#!/usr/bin/env python
"""Controller-level tests for Iterative Checkpoint Truncation (Flash-OPD).

These cover the two claims the method rests on, with no cluster and no GPU:

  1. Proposition 1 -- the retained length never falls below the true usable length E,
     for any rho, any window, any realization of the rate estimate.
  2. rho is a pure cost knob -- the retained length is *identical* across rho, while the
     number of verification rounds and the tokens decoded are not. This is the stronger
     statement, and it is what makes the sensitivity study in the paper interpretable:
     changing rho changes what the method costs, not what it supervises.

The oracle is synthetic rather than a recorded rollout, deliberately. phi -- one 0/1 per
position, 1 where top-k overlap falls below eta -- is the controller's *only* input, so
sweeping families of phi sequences (constant rates, bursts, rate collapse, degenerate
all-zero and all-one cases) exercises the state machine harder than any single recorded
trace, which is one realization of one pair at one step. Nothing here depends on data that
is not in this repository.

Run with `pytest tests/` or directly as a script.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

# Importable either from an editable install of ./verl or straight from the source tree.
_VERL = pathlib.Path(__file__).resolve().parents[1] / "verl"
if _VERL.is_dir() and str(_VERL) not in sys.path:
    sys.path.insert(0, str(_VERL))

from verl.utils.ict import ICTConfig, ICTController  # noqa: E402

B = 100
L = 4000


def phi_family(seed: int = 0) -> dict[str, np.ndarray]:
    """phi sequences spanning the regimes the controller has to survive.

    Rates are chosen so the first-passage time lands in very different places: `dense`
    crosses within a few hundred positions, `sparse` only near the end, and `never` does
    not cross at all.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {
        "dense": (rng.random(L) < 0.40).astype(int),
        "mid": (rng.random(L) < 0.12).astype(int),
        "sparse": (rng.random(L) < 0.03).astype(int),
        # rate collapses after position 500: extrapolation from the early window
        # overshoots badly, which is exactly the case Proposition 1 has to cover
        "collapse": np.concatenate([
            (rng.random(500) < 0.50).astype(int),
            (rng.random(L - 500) < 0.005).astype(int),
        ]),
        # rate only appears late: the early window estimates zero and the step diverges
        "late": np.concatenate([
            np.zeros(1500, dtype=int),
            (rng.random(L - 1500) < 0.30).astype(int),
        ]),
        # bursts, so no trailing window is representative of the sequence
        "bursty": np.concatenate([
            np.tile(np.concatenate([np.ones(40, dtype=int), np.zeros(360, dtype=int)]), 10),
        ]),
        "never": np.zeros(L, dtype=int),
        "immediate": np.ones(L, dtype=int),
    }
    assert all(len(v) == L for v in out.values())
    return out


def true_usable(phi: np.ndarray, allowance: int) -> int:
    """E = min{tau : c_tau >= allowance}, or the full length if it is never reached."""
    hit = np.nonzero(np.cumsum(phi) >= allowance)[0]
    return int(hit[0]) + 1 if len(hit) else len(phi)


def replay(phis: list[np.ndarray], cfg: ICTConfig) -> ICTController:
    """Drive the controller to completion against known phi arrays.

    Stands in for the online loop: whatever the controller asks to decode, it is handed the
    corresponding slice of the known phi. A slice shorter than requested is what the real
    loop produces when the sampler emits EOS or runs out of context.
    """
    ctl = ICTController(len(phis), cfg)
    while not ctl.done():
        ctl.observe({i: list(phis[i][ctl.seqs[i].verified:ctl.seqs[i].verified + d])
                     for i, d in ctl.deltas().items()})
    return ctl


RHOS = (1.0, 0.9, 0.8, 0.7, 0.5, 0.35)
WINDOWS = (64, 128, 256, 1024)
K0S = (64, 128, 1024)
KMINS = (1, 16, 512)


def test_proposition_1_no_early_truncation():
    """Retained length >= true usable length, swept over the whole knob space."""
    fam = phi_family()
    phis = list(fam.values())
    truth = [true_usable(p, B) for p in phis]
    checked = 0
    for rho in RHOS:
        for w in WINDOWS:
            for k0 in K0S:
                for kmin in KMINS:
                    cfg = ICTConfig(allowance=B, k0=k0, rho=rho, window=w,
                                    k_min=kmin, l_max=L)
                    got = replay(phis, cfg).final_lengths()
                    for name, g, t in zip(fam, got, truth):
                        assert g >= t, (
                            f"early truncation on {name}: retained {g} < usable {t} "
                            f"(rho={rho}, W={w}, k0={k0}, k_min={kmin})"
                        )
                    checked += len(got)
    assert checked == len(phis) * len(RHOS) * len(WINDOWS) * len(K0S) * len(KMINS)


def test_retained_length_is_exactly_the_first_passage_time():
    """Not merely >= E: the controller locates the crossing inside the verified block.

    `observe` scans the new positions and stops at the crossing rather than at the block
    end, so the retained length equals E exactly. Were it to stop at the block end, the
    retained length would depend on where checkpoints happened to land.
    """
    fam = phi_family()
    phis = list(fam.values())
    truth = [true_usable(p, B) for p in phis]
    for rho in RHOS:
        for w in WINDOWS:
            cfg = ICTConfig(allowance=B, k0=128, rho=rho, window=w, k_min=16, l_max=L)
            got = replay(phis, cfg).final_lengths()
            for name, g, t in zip(fam, got, truth):
                assert g == t, f"{name}: retained {g} != usable {t} (rho={rho}, W={w})"


def test_rho_changes_cost_not_supervision():
    """The point of the sensitivity study: rho moves rounds and decoding, not lengths."""
    phis = list(phi_family().values())
    baseline, seen = None, {}
    for rho in RHOS:
        cfg = ICTConfig(allowance=B, k0=128, rho=rho, window=128, k_min=16, l_max=L)
        ctl = replay(phis, cfg)
        lengths = ctl.final_lengths()
        if baseline is None:
            baseline = lengths
        assert lengths == baseline, f"rho={rho} changed the retained lengths"
        seen[rho] = (ctl.rounds, ctl.decoded_positions)

    # A smaller rho re-measures the rate sooner, so it cannot need fewer rounds, and the
    # extra caution has to show up as less overshoot (or at worst the same).
    assert seen[0.35][0] > seen[1.0][0], f"rho did not change the round count: {seen}"
    assert seen[0.35][1] <= seen[1.0][1], f"smaller rho decoded more tokens: {seen}"


def test_overshoot_is_decode_only():
    """Decoded positions may exceed what is retained; retained is never inflated."""
    phis = list(phi_family().values())
    cfg = ICTConfig(allowance=B, k0=128, rho=1.0, window=128, k_min=16, l_max=L)
    ctl = replay(phis, cfg)
    retained = sum(ctl.final_lengths())
    assert ctl.decoded_positions >= retained
    stats = ctl.stats()
    assert stats["ict/decode_overshoot"] >= 0.0
    # A checkpoint re-runs the prefix when the forward path cannot reuse a KV cache, so the
    # positions submitted to the teacher exceed the positions kept.
    assert stats["ict/verify_amplification"] >= 1.0


def test_degenerate_rates():
    """The two cases where the step rule has no usable rate estimate."""
    cfg = ICTConfig(allowance=B, k0=128, rho=1.0, window=128, k_min=16, l_max=L)

    # Never crosses: the rate estimate is 0, the extrapolated step diverges and is clamped
    # to l_max, and the trajectory retires holding everything it produced.
    ctl = replay([np.zeros(L, dtype=int)], cfg)
    assert ctl.final_lengths() == [L]
    assert ctl.stats()["ict/stop_context"] == 1.0

    # Every position fails: the crossing is at exactly `allowance`, inside the first block.
    ctl = replay([np.ones(L, dtype=int)], cfg)
    assert ctl.final_lengths() == [B]
    assert ctl.stats()["ict/stop_allowance"] == 1.0


def test_sampler_stopping_early_retains_everything():
    """A short phi list means EOS: the trajectory is complete, so it keeps what it has."""
    short = np.zeros(300, dtype=int)          # 300 positions, never reaches the allowance
    cfg = ICTConfig(allowance=B, k0=128, rho=1.0, window=128, k_min=16, l_max=L)
    ctl = replay([short], cfg)
    assert ctl.final_lengths() == [300]
    assert ctl.stats()["ict/stop_exhausted"] == 1.0


def test_l_max_is_the_one_cap_that_can_truncate_early():
    """L_max below the boundary stops a trajectory before E -- outside Proposition 1.

    G1 uses this deliberately. The test pins the behaviour so it stays visible rather than
    being mistaken for the method's own stopping rule.
    """
    phi = np.ones(L, dtype=int)               # boundary at exactly B
    cap = B // 2
    cfg = ICTConfig(allowance=B, k0=16, rho=1.0, window=16, k_min=16, l_max=cap)
    ctl = replay([phi], cfg)
    assert ctl.final_lengths() == [cap]
    assert ctl.final_lengths()[0] < true_usable(phi, B)
    assert ctl.stats()["ict/stop_context"] == 1.0


def test_higher_failure_rate_stops_earlier():
    """Ordering sanity: the boundary tracks the event rate monotonically."""
    rng = np.random.default_rng(0)
    phis = [(rng.random(L) < r).astype(int) for r in (0.02, 0.20, 0.50)]
    cfg = ICTConfig(allowance=20, k0=64, rho=0.7, window=128, k_min=16, l_max=L)
    got = replay(phis, cfg).final_lengths()
    assert got[0] > got[1] > got[2], got


def test_config_rejects_invalid_settings():
    for kwargs in (
        {"allowance": 0},
        {"rho": 0.0},
        {"rho": 1.5},
        {"k_min": 0},
        {"window": 0},
        {"k0": 0},
        {"k0": 99999, "l_max": 4096},
    ):
        try:
            ICTConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"ICTConfig accepted invalid settings: {kwargs}")


def _main() -> int:
    failed = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed.append(name)
            print(f"  FAIL  {name}\n        {exc}")
    print()
    if failed:
        print(f"FAILED: {len(failed)} test(s): {failed}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
