# Changes to the vendored verl

This repository ships a full copy of [verl](https://github.com/volcengine/verl) (version
`0.7.0.dev`, see `verl/verl/version/version`) rather than a patch, so that
`pip install -e ./verl` works against the tree you cloned. This file records what was
changed, so a reader can tell Flash-OPD apart from the framework it is built on.

There are two layers of modification, and it matters which is which:

- **Token-level on-policy distillation.** The `token_reward_direct` advantage estimator, the
  student-top-*k* candidate set, the teacher as a scoring model, and the `prune_opd` module
  come from [Prune-OPD](https://github.com/yangzhch6/Prune-OPD) (Yang et al., 2026) and, before
  that, from [THUNLP/OPD](https://github.com/thunlp/OPD). This repository is built on the
  Prune-OPD release and inherits that layer unchanged in substance; Prune-OPD is also one of
  the baselines Flash-OPD compares against.
- **Boundary verification.** The generate-then-verify loop, the per-trajectory stopping rule,
  and the per-request decode budgets are this paper's contribution.

The line counts below are measured against the Prune-OPD release, so they describe the second
layer only.

## New modules

| File | Lines | Responsibility |
| --- | --- | --- |
| `verl/verl/utils/ict.py` | 208 | `ICTController`: the stopping rule and the step rule. Pure Python -- no torch, no ray, no vLLM -- so it is unit-testable off a cluster (`tests/test_ict.py`). |
| `verl/verl/utils/ict_loop.py` | 209 | The tensor surgery around the loop: continuation batches, the compatibility indicator φ from the top-*k* overlap mask, and reassembly of a batch the post-loop stages cannot tell apart from a single-shot rollout. Tested by `tests/test_ict_loop.py`. |

The separation is deliberate: every decision that affects *what gets supervised* lives in
`ict.py` and is tested exhaustively; `ict_loop.py` only moves tensors, and its failure modes
are silent (an off-by-one in padding, or a row permutation between rounds, trains without
error but pairs each trajectory with another's supervision), so its invariants are asserted
one by one against hand-constructed cases.

## Modified files

| File | Δ | What changed |
| --- | --- | --- |
| `verl/verl/trainer/ppo/ray_trainer.py` | +527 | Orchestration. `_generate_sequences_ict` replaces the single `generate_sequences` call with the round loop; `_ict_verify` runs the teacher on the newly produced positions; `_ict_attach_cached_verify` feeds a round's verification outputs to the post-loop stages instead of recomputing them; `_set_per_sample_budgets_for_generation` installs the per-request decode budgets; `_ict_call_padded` keeps world-size divisibility as trajectories retire and the active set shrinks. |
| `verl/verl/workers/fsdp_workers.py` | +473 | The verification worker. Teacher forward over the new positions under a token budget (`ict_token_budget`), the rollout engine's residency policy across rounds (`ict_hold_awake`, `ict_reuse_weights`, `ict_sleep_rollout`), and round-boundary bookkeeping (`ict_first_round` / `ict_last_round`). Most of the churn is in the top-*k* log-prob path, which this restructures to run per round. |
| `verl/verl/utils/prune_opd.py` | +150 | The compatibility criterion: a position is a low-compatibility event when student-top-*k* / teacher-top-*k* overlap falls below η. Also derives the reward attenuation **off** under Flash-OPD -- `attenuate = not ict_on` -- see the note below. |
| `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | +41 | Per-request decode budgets (`psb_budget`). Each trajectory asks the engine for its own number of tokens, so a round is not bounded by its longest member. |
| `verl/verl/workers/actor/dp_actor.py` | +24 | Top-*k* log-prob path, so verification returns the candidate overlap the criterion needs rather than full-vocabulary logits. |
| `verl/verl/utils/profiler/performance.py` | +18 | `VERL_SKIP_TIMING_ALLGATHER`. The timing all-gather fires once per profiled region, which under a multi-round step is a per-round collective on the critical path. |

## What ships but does not run

Two Prune-OPD mechanisms remain in the tree and are **inert** in every configuration in
`experiments_scripts/`:

- **Reward attenuation** (`w_drop`, `w_base`). `prune_opd.py` sets `attenuate = not ict_on`, so
  with `prune_opd.ict.enable=True` neither value reaches the loss and every retained position
  carries weight 1.0. `w_drop` still scales the logged `E` diagnostic -- the position count a
  1/`w_drop` reliability budget would have permitted -- which is why the launchers pin it
  rather than leave it implicit.
- **`dynamic_response_length`**, the batch-shared horizon controller. The Flash-OPD launchers
  pass `dynamic_response_length.enable=False`; the OPD baselines pass no `prune_opd` subtree at
  all, so it is never constructed.

Neither was stripped. `prune_opd.py`'s boundary-verification branch shares the module with
them, and removing them would perturb the orchestration path in `ray_trainer.py` that the
reported runs went through, for no benefit to a reader.

## The OPD baseline

There is no flag that turns Flash-OPD off. The OPD baseline is `run_opd` invoked with no
`actor_rollout_ref.rollout.prune_opd` overrides: the config subtree is never created, every
guard reads as disabled, and the trainer takes the stock single-shot `generate_sequences` path
with full-horizon decoding. `DRY_RUN=1 bash experiments_scripts/opd-g2-*.sh` shows this
directly -- the printed command contains no `prune_opd` key.
