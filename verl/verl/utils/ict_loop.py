"""Training-loop integration for Iterative Checkpoint Truncation.

Kept in its own module so `ray_trainer.fit()` gains one guarded call rather than an
inlined loop, and so the tensor surgery here can be tested against the controller in
`verl.utils.ict` without a cluster.

WHAT THIS HAS TO DO, and why each piece is awkward:

* vLLM must be asked to *continue* a trajectory. Its rollout entry point takes a padded
  `input_ids` tensor and derives `raw_prompt_ids` by stripping left padding, so the way
  to continue is to hand it a batch whose prompt is `original_prompt + generated_so_far`.
  With prefix caching on, the shared prefix is served from cache rather than re-prefilled.

* phi must be computed on the newly produced positions only. The overlap mask comes from
  the teacher, and the teacher's top-k comparison needs the student's top-k, so each round
  costs one student log-prob pass plus one teacher pass. That is the dominant cost of the
  method and is what the smoke test measures.

* Row order must survive every round. `_balance_batch` re-sorts by sequence length, so it
  MUST NOT run inside the checkpoint loop -- only after the final lengths are known.
  Rounds are matched to controller state by position, so any reordering silently pairs a
  trajectory with another's failure count.
"""

from __future__ import annotations

import numpy as np
import torch

from verl.protocol import DataProto
from verl.utils.ict import ICTConfig, ICTController


def ict_config_from_cfg(cfg) -> ICTConfig:
    """Build an ICTConfig from the `prune_opd.ict` config subtree."""
    get = (lambda k, d: cfg.get(k, d)) if cfg is not None else (lambda k, d: d)
    return ICTConfig(
        allowance=int(get("allowance", 200)),
        k0=int(get("k0", 128)),
        rho=float(get("rho", 0.7)),
        window=int(get("window", 128)),
        k_min=int(get("k_min", 16)),
        l_max=int(get("l_max", 12288)),
    )


def _strip_left_pad(row: torch.Tensor, pad_id: int) -> list[int]:
    """Left-padded prompt row -> token id list. Mirrors vLLM's _pre_process_inputs."""
    nz = torch.nonzero(row != pad_id, as_tuple=False)
    if nz.numel() == 0:
        return []
    return row[nz[0][0]:].tolist()


def build_continuation_batch(
    gen_batch: DataProto,
    prompt_ids: list[list[int]],
    grown: list[list[int]],
    active: list[int],
    deltas: dict[int, int],
    pad_id: int,
    device,
) -> DataProto:
    """Assemble the round's generation request for the active trajectories.

    `prompt_ids[i]` is trajectory i's original prompt; `grown[i]` the tokens generated for
    it so far. The request's prompt is their concatenation, so vLLM continues rather than
    restarts. Left padding is applied to a common length, matching what the rollout worker
    expects, and `raw_prompt_ids` is supplied explicitly so it does not have to re-derive
    it from the padded tensor.
    """
    seqs = [prompt_ids[i] + grown[i] for i in active]
    width = max(len(s) for s in seqs)
    n = len(seqs)

    input_ids = torch.full((n, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, width), dtype=torch.long)
    position_ids = torch.zeros((n, width), dtype=torch.long)
    for r, s in enumerate(seqs):
        L = len(s)
        input_ids[r, width - L:] = torch.tensor(s, dtype=torch.long)
        attention_mask[r, width - L:] = 1
        position_ids[r, width - L:] = torch.arange(L, dtype=torch.long)

    out = DataProto.from_dict(
        tensors={
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "position_ids": position_ids.to(device),
        }
    )
    out.non_tensor_batch["raw_prompt_ids"] = np.array([list(s) for s in seqs], dtype=object)
    out.meta_info = dict(gen_batch.meta_info)
    # Per-request caps: each trajectory decodes only up to its own next checkpoint.
    out.batch["psb_budget"] = torch.tensor(
        [max(1, deltas[i]) for i in active], dtype=torch.long, device=device
    )
    # The scalar cap must not bind before the per-request caps do.
    out.meta_info["max_response_length"] = int(max(deltas[i] for i in active))
    return out


def new_token_counts(response_mask: torch.Tensor, requested: list[int]) -> list[int]:
    """How many tokens each row actually produced this round.

    `pad_token_id == eos_token_id` on these tokenizers, so a padded tail and a real EOS
    are the same token id and cannot be told apart by inspection. verl's
    `get_response_mask` keeps the first such token as valid, which is correct when the
    model emitted EOS but over-counts by exactly one when the row instead ran into its
    cap: 16 real tokens followed by padding yields a mask sum of 17.

    Clamping to the number of tokens requested resolves both cases. If the row stopped
    early by emitting EOS, the mask sum is at most `requested` and is used as is; if it
    ran to the cap, the sum is `requested + 1` and the clamp discards the phantom
    position. Nothing else in the loop needs to know which of the two happened.
    """
    sums = response_mask.sum(dim=-1).tolist()
    return [min(int(s), int(r)) for s, r in zip(sums, requested)]


def extract_new_tokens(round_out: DataProto, counts: list[int]) -> list[list[int]]:
    """Freshly generated token ids per row, truncated to `counts`.

    The slicing is done on a single host-side copy rather than per row. `t[r][:c].tolist()`
    on a CUDA tensor forces a device synchronisation for every row, so the cost scales with
    the row count rather than with the data; moving the whole tensor once and slicing in
    numpy removes that. Same values either way -- this is a transfer pattern, not a change
    in what is computed.
    """
    resp = round_out.batch["responses"]
    host = resp.detach().cpu().numpy()
    return [host[r, : counts[r]].tolist() for r in range(host.shape[0])]


def assemble_final_batch(
    gen_batch: DataProto,
    prompt_ids: list[list[int]],
    grown: list[list[int]],
    lengths: list[int],
    pad_id: int,
    eos_id,
    response_width: int,
    device,
) -> DataProto:
    """Build the batch the rest of the step consumes, with each trajectory truncated.

    Produces exactly the tensor layout the single-shot rollout path produces --
    left-padded prompt, right-padded response, concatenated `input_ids`, and matching
    masks -- so every downstream stage (log-prob, teacher, advantage, update) is unchanged.
    """
    n = len(prompt_ids)
    prompt_width = max(len(p) for p in prompt_ids)

    prompts_t = torch.full((n, prompt_width), pad_id, dtype=torch.long)
    prompt_mask = torch.zeros((n, prompt_width), dtype=torch.long)
    prompt_pos = torch.zeros((n, prompt_width), dtype=torch.long)
    resp_t = torch.full((n, response_width), pad_id, dtype=torch.long)
    resp_mask = torch.zeros((n, response_width), dtype=torch.long)

    for r in range(n):
        p = prompt_ids[r]
        prompts_t[r, prompt_width - len(p):] = torch.tensor(p, dtype=torch.long)
        prompt_mask[r, prompt_width - len(p):] = 1
        prompt_pos[r, prompt_width - len(p):] = torch.arange(len(p), dtype=torch.long)
        # truncate to the length the controller retained
        keep = grown[r][: lengths[r]][:response_width]
        if keep:
            resp_t[r, : len(keep)] = torch.tensor(keep, dtype=torch.long)
            resp_mask[r, : len(keep)] = 1

    delta = torch.arange(1, response_width + 1).unsqueeze(0).expand(n, -1)
    resp_pos = prompt_pos[:, -1:] + delta

    tensors = {
        "prompts": prompts_t.to(device),
        "responses": resp_t.to(device),
        "input_ids": torch.cat([prompts_t, resp_t], dim=-1).to(device),
        "attention_mask": torch.cat([prompt_mask, resp_mask], dim=-1).to(device),
        "position_ids": torch.cat([prompt_pos, resp_pos], dim=-1).to(device),
    }
    out = DataProto.from_dict(tensors=tensors)
    out.meta_info = dict(gen_batch.meta_info)
    for k, v in gen_batch.non_tensor_batch.items():
        out.non_tensor_batch[k] = v
    return out


def phi_from_overlap(
    overlap_mask: torch.Tensor,
    response_mask: torch.Tensor,
    threshold: float,
    top_k: int,
) -> list[list[int]]:
    """Reliability indicators per row: 1 where top-k overlap falls below `threshold`.

    `overlap_mask` is (batch, resp_len, k) with 1 marking a student candidate that also
    appears in the teacher's top-k, so its mean over the last axis is the overlap ratio.
    Only positions inside `response_mask` are returned; padding carries no indicator.

    The reduction stays on device and the result crosses to the host exactly once. The
    obvious per-row form, `bad[r][valid].tolist()`, synchronises once per row, which at
    n=256 and L=3400 measured 2.56 s per step against 0.05 s here -- 50x, and it grows with
    the batch because every row pays a separate round trip. The indicators are identical.
    """
    ratio = overlap_mask.float().mean(dim=-1)
    bad = (ratio < threshold).to(torch.uint8)
    bad_host = bad.detach().cpu().numpy()
    valid_host = response_mask.detach().cpu().numpy().astype(bool)
    return [bad_host[r][valid_host[r]].tolist() for r in range(bad_host.shape[0])]
