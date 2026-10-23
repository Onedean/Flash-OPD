#!/usr/bin/env python
"""Tests for the ICT tensor surgery (verl/verl/utils/ict_loop.py).

The failure modes here are silent: an off-by-one in left/right padding, or a row
permutation between rounds, produces a batch that trains without error but pairs each
trajectory with another's supervision. So every invariant is checked explicitly against
hand-constructed cases rather than eyeballed on a real run.

Needs torch but no GPU and no model. Run with `pytest tests/` or directly as a script.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

# Importable either from an editable install of ./verl or straight from the source tree.
_VERL = pathlib.Path(__file__).resolve().parents[1] / "verl"
if _VERL.is_dir() and str(_VERL) not in sys.path:
    sys.path.insert(0, str(_VERL))

from verl.protocol import DataProto  # noqa: E402
from verl.utils.ict import ICTConfig, ICTController  # noqa: E402
from verl.utils.ict_loop import (  # noqa: E402
    assemble_final_batch,
    build_continuation_batch,
    extract_new_tokens,
    new_token_counts,
    phi_from_overlap,
)

PAD = 0
EOS = 99


# --- A: continuation batch -------------------------------------------------------------

def test_continuation_batch_makes_prompt_plus_generated_the_new_prompt():
    prompt_ids = [[11, 12, 13], [21, 22], [31, 32, 33, 34]]
    grown = [[41, 42], [], [51]]
    active = [0, 1, 2]
    deltas = {0: 64, 1: 128, 2: 16}
    gen = DataProto.from_dict(tensors={"input_ids": torch.zeros(3, 4, dtype=torch.long)})
    gen.meta_info = {"eos_token_id": EOS}

    cb = build_continuation_batch(gen, prompt_ids, grown, active, deltas, PAD, "cpu")

    raw = cb.non_tensor_batch["raw_prompt_ids"]
    assert list(raw[0]) == [11, 12, 13, 41, 42], list(raw[0])
    assert list(raw[1]) == [21, 22], list(raw[1])
    assert list(raw[2]) == [31, 32, 33, 34, 51], list(raw[2])

    ii, am = cb.batch["input_ids"], cb.batch["attention_mask"]
    assert ii.size(1) == 5, f"width should be the longest sequence, got {ii.size(1)}"
    # row 1 is the shortest (len 2), so it carries 3 pads on the LEFT. Right-padding here
    # would put the pads between the prefix and the tokens about to be generated.
    assert ii[1].tolist() == [PAD, PAD, PAD, 21, 22], ii[1].tolist()
    assert am[1].tolist() == [0, 0, 0, 1, 1], am[1].tolist()
    assert cb.batch["position_ids"][1].tolist() == [0, 0, 0, 0, 1], \
        cb.batch["position_ids"][1].tolist()

    # Per-request caps: each trajectory asks the engine for its own number of tokens.
    assert cb.batch["psb_budget"].tolist() == [64, 128, 16], cb.batch["psb_budget"].tolist()
    # The scalar cap is the max of them, so it never binds before the per-request caps do.
    assert cb.meta_info["max_response_length"] == 128, cb.meta_info["max_response_length"]


# --- B: token counts under pad_token_id == eos_token_id --------------------------------

SHARED = 151643


def _shared_pad_eos_case():
    """A response block where the pad token and EOS are the same id.

    The real tokenizers have pad_token_id == eos_token_id == 151643, so a padded tail and
    a genuine EOS are indistinguishable. The response mask keeps the first such token as
    valid, which is right when the model emitted EOS but over-counts by exactly one when
    the row ran into its cap.
    """
    resp = torch.tensor([
        [10, 11, 12, 13, SHARED, SHARED],   # ran into a 4-token cap
        [20, 21, 22, 23, 24, 25],           # filled a 6-token request exactly
        [30, 31, SHARED, SHARED, SHARED, SHARED],   # genuine early stop
    ])
    eos_mask = torch.isin(resp, torch.tensor([SHARED])).int()
    rmask = (eos_mask.cumsum(dim=1) - eos_mask).eq(0).int()
    return resp, rmask


def test_response_mask_overcounts_a_capped_row():
    """The off-by-one this clamp exists to absorb: a 4-token request yields 5 indicators."""
    _, rmask = _shared_pad_eos_case()
    assert rmask[0].sum().item() == 5, rmask.sum(dim=1).tolist()
    assert rmask[1].sum().item() == 6, rmask.sum(dim=1).tolist()


def test_new_token_counts_clamps_to_the_request():
    _, rmask = _shared_pad_eos_case()
    requested = [4, 6, 9]
    counts = new_token_counts(rmask, requested)
    assert counts[0] == 4, f"capped row not clamped: {counts}"
    assert counts[1] == 6, f"exact-fill row changed: {counts}"
    assert counts[2] == 3, f"early stop should keep its real length: {counts}"
    # The invariant the controller relies on: observe() rejects more indicators than it
    # asked for, so no count may ever exceed its request.
    assert all(c <= r for c, r in zip(counts, requested)), f"{counts} vs {requested}"


def test_extract_new_tokens_matches_the_counts():
    resp, rmask = _shared_pad_eos_case()
    counts = new_token_counts(rmask, [4, 6, 9])
    new = extract_new_tokens(DataProto.from_dict(tensors={"responses": resp}), counts)
    assert [len(x) for x in new] == counts, [len(x) for x in new]
    assert new[0] == [10, 11, 12, 13], new[0]


def test_phi_clipped_to_counts_fits_every_request():
    _, rmask = _shared_pad_eos_case()
    requested = [4, 6, 9]
    counts = new_token_counts(rmask, requested)
    phi_b = phi_from_overlap(torch.zeros(3, 6, 4), rmask, threshold=0.7, top_k=4)
    clipped = [list(p)[:c] for p, c in zip(phi_b, counts)]
    assert all(len(c) <= r for c, r in zip(clipped, requested)), \
        f"{[len(c) for c in clipped]} vs {requested}"


# --- C: phi from the overlap mask ------------------------------------------------------

def test_phi_from_overlap_counts_valid_positions_only():
    # 2 rows, 4 positions, k=4. The overlap ratio is the mean over the last axis.
    om = torch.zeros(2, 4, 4)
    om[0, 0] = torch.tensor([1., 1., 1., 1.])   # ratio 1.00 -> compatible
    om[0, 1] = torch.tensor([1., 0., 0., 0.])   # ratio 0.25 -> low-compatibility event
    om[0, 2] = torch.tensor([1., 1., 1., 0.])   # ratio 0.75 -> compatible at eta=0.7
    om[0, 3] = torch.tensor([0., 0., 0., 0.])   # ratio 0.00 -> event, but MASKED OUT below
    om[1, 0] = torch.tensor([1., 1., 0., 0.])   # ratio 0.50 -> event
    rm = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]])

    phi = phi_from_overlap(om, rm, threshold=0.7, top_k=4)

    assert phi[0] == [0, 1, 0], phi[0]
    assert phi[1] == [1], phi[1]
    # Padding must not be charged against the allowance: the masked position at [0, 3]
    # has ratio 0.0 and would otherwise read as an event.
    assert len(phi[0]) == 3 and len(phi[1]) == 1, phi


# --- D: final batch --------------------------------------------------------------------

def test_final_batch_layout_matches_the_single_shot_path():
    """The post-loop stages must not be able to tell a multi-round batch apart."""
    prompt_ids = [[11, 12, 13], [21, 22]]
    grown = [[41, 42, 43, 44], [51, 52, 53]]
    lengths = [2, 3]        # what the controller retained, below what was generated
    gen = DataProto.from_dict(tensors={"input_ids": torch.zeros(2, 3, dtype=torch.long)})
    gen.meta_info = {"eos_token_id": EOS}
    gen.non_tensor_batch["uid"] = np.array(["a", "b"], dtype=object)

    fb = assemble_final_batch(gen, prompt_ids, grown, lengths, PAD, EOS,
                              response_width=6, device="cpu")

    assert fb.batch["prompts"][1].tolist() == [PAD, 21, 22], \
        f"prompt must stay left-padded: {fb.batch['prompts'][1].tolist()}"
    assert fb.batch["responses"][0].tolist() == [41, 42, PAD, PAD, PAD, PAD], \
        f"response must be right-padded and truncated: {fb.batch['responses'][0].tolist()}"
    assert fb.batch["responses"][1].tolist() == [51, 52, 53, PAD, PAD, PAD], \
        f"truncation must honour the per-row length: {fb.batch['responses'][1].tolist()}"
    assert fb.batch["input_ids"][0].tolist() == [11, 12, 13, 41, 42, PAD, PAD, PAD, PAD], \
        fb.batch["input_ids"][0].tolist()

    am = fb.batch["attention_mask"]
    assert am[0].tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0], am[0].tolist()
    assert am[1].tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0], \
        f"row 1 must account for its prompt pad: {am[1].tolist()}"
    assert fb.batch["position_ids"][0].tolist()[:6] == [0, 1, 2, 3, 4, 5], \
        fb.batch["position_ids"][0].tolist()
    assert list(fb.non_tensor_batch["uid"]) == ["a", "b"], "non_tensor_batch was not carried"


# --- E: end to end ---------------------------------------------------------------------

def test_controller_drive_preserves_row_order():
    """Three trajectories with deliberately different event rates.

    The rates are distinct enough that a row permutation between rounds would change the
    retained lengths, so the equality against the offline first-passage times is also a
    row-order check.
    """
    rng = np.random.default_rng(0)
    truth_phi = [(rng.random(4000) < r).astype(int) for r in (0.02, 0.20, 0.50)]
    B = 20
    cfg = ICTConfig(allowance=B, k0=64, rho=0.7, window=128, k_min=16, l_max=4000)

    ctl = ICTController(3, cfg)
    while not ctl.done():
        ctl.observe({i: list(truth_phi[i][ctl.seqs[i].verified:ctl.seqs[i].verified + d])
                     for i, d in ctl.deltas().items()})

    got = ctl.final_lengths()
    expect = []
    for p in truth_phi:
        hit = np.nonzero(np.cumsum(p) >= B)[0]
        expect.append(int(hit[0]) + 1 if len(hit) else len(p))

    assert got == expect, f"got={got} expect={expect}"
    assert got[0] > got[1] > got[2], f"higher event rate should stop earlier: {got}"

    st = ctl.stats()
    assert st["ict/stop_allowance"] == 3.0, st
    assert st["ict/verify_amplification"] > 1.0, st["ict/verify_amplification"]


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
