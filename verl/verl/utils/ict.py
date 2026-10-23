"""Iterative Checkpoint Truncation (ICT) controller.

ICT replaces horizon *prediction* with sequential *verification*. The usable length
of a trajectory is the first position at which its accumulated reliability-failure
count reaches an allowance B (a first-passage time). That count is a function of the
prefix alone, so during generation it can be read exactly rather than estimated.

This module holds only the controller state machine -- no torch, no ray, no vLLM --
so the step rule can be unit-tested and replayed against offline trajectory dumps
without a cluster. The trainer drives it:

    ctl = ICTController(n, cfg)
    while not ctl.done():
        targets = ctl.targets()          # per-sequence position to decode to
        ... decode active sequences to targets ...
        ... teacher forward on the NEW positions only ...
        ctl.observe(phi)                 # phi[i] = list of 0/1 for the new positions
    lengths = ctl.final_lengths()

Design note on the stopping rule: `observe` locates the boundary inside the newly
verified block rather than at the block's end. Stopping at the block end would make
the retained length depend on where checkpoints happened to fall, so two runs with
different rho would truncate the same trajectory at different places; locating the
crossing keeps `stop_at` equal to the true first-passage time whenever the crossing
is inside a verified block, which is what makes rho a pure cost knob.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ICTConfig:
    allowance: int = 200          # B: unreliable positions a trajectory may contain
    k0: int = 128                 # first checkpoint
    rho: float = 0.7              # step discount; <1 re-measures the rate before the boundary
    window: int = 128             # W: trailing window for the rate estimate
    k_min: int = 16               # minimum advance, guards against a vanishing step
    l_max: int = 12288            # context limit; also absorbs the lambda_hat == 0 case

    def __post_init__(self):
        if self.allowance < 1:
            raise ValueError(f"allowance must be >= 1, got {self.allowance}")
        if not (0.0 < self.rho <= 1.0):
            raise ValueError(f"rho must be in (0, 1], got {self.rho}")
        if self.k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {self.k_min}")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.k0 < 1 or self.k0 > self.l_max:
            raise ValueError(f"k0 must be in [1, l_max], got {self.k0}")


@dataclass
class _Seq:
    """Per-trajectory state. Nothing here is shared across trajectories."""
    target: int                      # position this trajectory is being decoded to
    verified: int = 0                # positions whose phi is known
    count: int = 0                   # c: accumulated failures over [0, verified)
    active: bool = True
    stop_at: int | None = None       # final retained length once inactive
    # phi history, needed for the trailing-window rate. Kept as a flat list because
    # the window is small and trajectories are short relative to the batch.
    phi: list[int] = field(default_factory=list)
    reason: str = ""                 # allowance | context | exhausted


class ICTController:
    """Drives the ICT loop for one training step over `n` trajectories."""

    def __init__(self, n: int, cfg: ICTConfig):
        self.cfg = cfg
        self.n = n
        self.seqs = [_Seq(target=min(cfg.k0, cfg.l_max)) for _ in range(n)]
        self.rounds = 0
        # Cost accounting, kept separate because the two are not the same quantity:
        #   decoded_positions -- each position produced once. Drives decode cost.
        #   prefix_positions  -- sum of checkpoint positions. A teacher call at k must
        #                        re-run [0, k) whenever the forward path cannot reuse a
        #                        KV cache, so verification cost scales with THIS, not
        #                        with the number of unique positions.
        self.decoded_positions = 0
        self.prefix_positions = 0

    # ---- driving ----------------------------------------------------------

    def done(self) -> bool:
        return not any(s.active for s in self.seqs)

    def active_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.seqs) if s.active]

    def targets(self) -> dict[int, int]:
        """Absolute position each active trajectory should be decoded to."""
        return {i: self.seqs[i].target for i, s in enumerate(self.seqs) if s.active}

    def deltas(self) -> dict[int, int]:
        """Additional tokens to request per active trajectory (target - verified)."""
        return {i: self.seqs[i].target - self.seqs[i].verified
                for i, s in enumerate(self.seqs) if s.active}

    def observe(self, phi_new: dict[int, list[int]]) -> None:
        """Accept the reliability indicators for each active trajectory's new positions.

        phi_new[i] holds one 0/1 per position in (verified, target]. A short list means
        the sampler produced fewer tokens than requested (EOS or the context limit), and
        that trajectory is finished: there is nothing further to decode, so it retains
        what it has regardless of whether the allowance was reached.
        """
        self.rounds += 1
        for i, phi in phi_new.items():
            s = self.seqs[i]
            if not s.active:
                raise RuntimeError(f"observe() got data for inactive trajectory {i}")
            requested = s.target - s.verified
            produced = len(phi)
            if produced > requested:
                raise ValueError(
                    f"trajectory {i}: got {produced} indicators for {requested} requested positions"
                )

            # locate the crossing INSIDE this block, so the retained length is the true
            # first-passage time and does not depend on where the checkpoint landed
            crossing = None
            c = s.count
            for j, p in enumerate(phi):
                c += int(p)
                if c >= self.cfg.allowance and crossing is None:
                    crossing = s.verified + j + 1
            s.phi.extend(int(p) for p in phi)
            s.count = c
            s.verified += produced
            self.decoded_positions += produced
            self.prefix_positions += s.verified      # this call re-ran [0, verified)

            if crossing is not None:
                self._retire(s, crossing, "allowance")
            elif produced < requested:
                # sampler stopped early: sequence is complete, keep all of it
                self._retire(s, s.verified, "exhausted")
            elif s.verified >= self.cfg.l_max:
                self._retire(s, s.verified, "context")
            else:
                s.target = self._next_target(s)

    # ---- the step rule ---------------------------------------------------

    def _rate(self, s: _Seq) -> float:
        """Failure rate over the trailing window of W positions."""
        w = min(self.cfg.window, len(s.phi))
        if w == 0:
            return 0.0
        return sum(s.phi[-w:]) / w

    def _next_target(self, s: _Seq) -> int:
        """k <- min( k + max(k_min, ceil(rho * (B - c) / lambda_hat)), l_max ).

        lambda_hat == 0 makes the extrapolated distance diverge; the l_max clamp absorbs
        that case, which is the intended behaviour -- no failure in the trailing window is
        evidence the boundary is not near.
        """
        remaining = self.cfg.allowance - s.count
        rate = self._rate(s)
        if rate <= 0.0:
            advance = self.cfg.l_max                     # diverges, clamped below
        else:
            import math
            advance = math.ceil(self.cfg.rho * remaining / rate)
        advance = max(self.cfg.k_min, advance)
        return min(s.verified + advance, self.cfg.l_max)

    def _retire(self, s: _Seq, stop_at: int, reason: str) -> None:
        s.active = False
        s.stop_at = stop_at
        s.reason = reason

    # ---- results ---------------------------------------------------------

    def final_lengths(self) -> list[int]:
        out = []
        for i, s in enumerate(self.seqs):
            if s.stop_at is None:
                raise RuntimeError(f"trajectory {i} never retired")
            out.append(s.stop_at)
        return out

    def stats(self) -> dict[str, float]:
        lens = [s.stop_at for s in self.seqs if s.stop_at is not None]
        n = max(1, len(lens))
        reasons = {}
        for s in self.seqs:
            reasons[s.reason] = reasons.get(s.reason, 0) + 1
        retained = sum(lens)
        return {
            "ict/rounds": float(self.rounds),
            "ict/len_mean": sum(lens) / n,
            "ict/len_min": float(min(lens)) if lens else 0.0,
            "ict/len_max": float(max(lens)) if lens else 0.0,
            # tokens decoded but not retained: the price of overshooting the boundary
            "ict/decode_overshoot": (self.decoded_positions - retained) / n,
            # positions submitted to the teacher / positions kept. >1 because a checkpoint
            # re-runs the prefix when the forward path cannot reuse a KV cache.
            "ict/verify_amplification": self.prefix_positions / max(1, retained),
            "ict/stop_allowance": float(reasons.get("allowance", 0)),
            "ict/stop_context": float(reasons.get("context", 0)),
            "ict/stop_exhausted": float(reasons.get("exhausted", 0)),
        }
