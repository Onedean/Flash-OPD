from __future__ import annotations

import math
from typing import Any

import torch


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _local_info_gain(
    student_top_k_log_probs: torch.Tensor,
    teacher_top_k_log_probs: torch.Tensor,
    overlap_mask: torch.Tensor | None,
    restrict_to_overlap: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Per-position KL(student || teacher) over the student's top-k candidates.

    Both inputs are aligned on the student's top-k token ids, so this is a local
    measure of how much the teacher still disagrees with the student at a state the
    student actually visited -- i.e. how much there is left to learn here. When
    restrict_to_overlap is set the KL is computed on the intersection of the two
    top-k supports and renormalized there, matching the overlap-token formulation
    used in the OPD-dynamics literature; otherwise it uses the full student top-k.

    Returns a (batch, seqlen) tensor in nats.
    """
    p = student_top_k_log_probs.float().exp()
    q = teacher_top_k_log_probs.float().exp()

    if restrict_to_overlap and overlap_mask is not None:
        m = overlap_mask.to(p.dtype)
        p = p * m
        q = q * m

    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (p * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(dim=-1)


def apply_prune_opd_to_scores(
    rm_scores: torch.Tensor,
    overlap_mask: torch.Tensor | None,
    response_mask: torch.Tensor | None,
    config: Any,
    responses: torch.Tensor | None = None,
    teacher_top_k_ids: torch.Tensor | None = None,
    teacher_top_k_log_probs: torch.Tensor | None = None,
    student_top_k_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply prune-opd weights to OPD token-level scores.

    prune-opd builds a monotone non-increasing token weight from a teacher
    reliability signal. Every violation decays all future OPD weights by a
    fixed step.

    When ICT drives generation the decay is disabled (see `attenuate` below): ICT already
    stops each trajectory at its own boundary, so there is no past-the-boundary tail for a
    weight to suppress, and the reliability signal is retained only as a diagnostic.
    """

    enabled = bool(_cfg_get(config, "enable", False))
    if not enabled:
        return rm_scores, {}

    metric = _cfg_get(config, "metric", "overlap_ratio")

    if response_mask is None:
        raise ValueError("prune-opd requires response_mask, but it is missing from the OPD batch.")
    if rm_scores.dim() != 3:
        raise ValueError(f"prune-opd expects 3D rm_scores for top-k OPD, got shape {tuple(rm_scores.shape)}.")

    threshold = float(_cfg_get(config, "threshold", 0.9))
    w_drop = float(_cfg_get(config, "w_drop", 0.02))
    w_base = float(_cfg_get(config, "w_base", 0.0))
    positive_eps = float(_cfg_get(config, "positive_eps", 1e-8))
    # Positional attenuation exists to suppress supervision past the reliability boundary.
    # Under ICT that boundary IS the retained length, so the decay is not merely redundant
    # but actively harmful, and it must default to off whenever ICT drives generation --
    # `attenuate` is derived from the ict subtree rather than passed in, so a launcher
    # cannot enable one without the other. `attenuate=False` also makes w_drop inert, which
    # is what ict-run.sh already documents.
    ict_cfg = _cfg_get(config, "ict", None)
    ict_on = ict_cfg is not None and bool(_cfg_get(ict_cfg, "enable", False))
    attenuate = bool(_cfg_get(config, "attenuate", not ict_on))
    if not attenuate and w_base != 0.0:
        # With no decay every retained position already sits at 1.0, so a floor cannot lift
        # anything -- it only rescales every weight to 1 + w_base, which under token-mean
        # aggregation is an unannounced learning-rate multiplier. Refuse unless the caller
        # says that scaling IS the thing being measured.
        #
        # The opt-in exists for one specific ablation. sae-opd.md 4.20 reports w_base 0 -> 0.5
        # worth +2.69 macro on G3 (n=3, sd 0.03) and reads it as "the beyond-E tail carries
        # signal". But on G3 that change also multiplies the mean loss weight by roughly 6x,
        # because ~80% of positions sit at weight 0 with w_base=0. The two explanations are
        # confounded in that experiment. Running ICT at a uniform 1.5 separates them: ICT has
        # no beyond-boundary tail at all, so any gain here is scale, not signal.
        if not bool(_cfg_get(config, "allow_uniform_w_base", False)):
            raise ValueError(
                f"prune-opd: w_base must be 0 when attenuation is off (got {w_base}); "
                "a floor weight has no effect on an unattenuated weight beyond scaling the "
                "loss. Set prune_opd.allow_uniform_w_base=True if that scaling is the "
                "quantity under test."
            )
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"prune-opd threshold must be in (0, 1], got {threshold}.")
    if not math.isfinite(w_drop) or w_drop < 0.0:
        raise ValueError(f"prune-opd w_drop must be finite and non-negative, got {w_drop}.")
    if not math.isfinite(w_base) or w_base < 0.0:
        raise ValueError(f"prune-opd w_base must be finite and non-negative, got {w_base}.")

    device = rm_scores.device
    response_mask = response_mask.to(device)

    valid_mask = response_mask.bool()
    aux_tensors = {}

    if metric == "overlap_ratio":
        if overlap_mask is None:
            raise ValueError("prune-opd requires overlap_mask, but it is missing from the OPD batch.")

        overlap_mask = overlap_mask.to(device)
        overlap_ratio = overlap_mask.float().mean(dim=-1)
        bad_event = (overlap_ratio < threshold) & valid_mask
        aux_tensors["prune_opd_overlap_ratio"] = overlap_ratio
    elif metric == "teacher_top_p_accept":
        if responses is None:
            raise ValueError("prune-opd metric teacher_top_p_accept requires responses.")
        if teacher_top_k_ids is None:
            raise ValueError("prune-opd metric teacher_top_p_accept requires teacher_top_k_ids.")
        if teacher_top_k_log_probs is None:
            raise ValueError("prune-opd metric teacher_top_p_accept requires teacher_top_k_log_probs.")

        responses = responses.to(device)
        teacher_top_k_ids = teacher_top_k_ids.to(device)
        teacher_top_k_log_probs = teacher_top_k_log_probs.to(device)

        teacher_top_k_probs = teacher_top_k_log_probs.float().exp()
        teacher_top_k_cum_probs = teacher_top_k_probs.cumsum(dim=-1)
        crossed = teacher_top_k_cum_probs >= threshold
        has_crossed = crossed.any(dim=-1, keepdim=True)
        first_cross_idx = crossed.to(torch.long).argmax(dim=-1, keepdim=True)
        rank_idx = torch.arange(teacher_top_k_ids.size(-1), device=device).view(1, 1, -1)
        top_p_mask = rank_idx <= first_cross_idx
        accept_mask = torch.where(has_crossed, top_p_mask, torch.ones_like(top_p_mask, dtype=torch.bool))

        response_in_accept_set = ((teacher_top_k_ids == responses.unsqueeze(-1)) & accept_mask).any(dim=-1)
        bad_event = (~response_in_accept_set) & valid_mask
        aux_tensors["prune_opd_teacher_top_p_accept"] = response_in_accept_set.to(torch.float32)
    elif metric == "ig_gated":
        # Two orthogonal questions, kept separate instead of collapsed into one scalar:
        #   is there anything left to learn here?  -> local KL (info gain)
        #   is the teacher's advice trustworthy?   -> top-k overlap (Prune-OPD's signal)
        # Measured corr(overlap, info_gain) = -0.095 on 2.0M student-visited positions, i.e.
        # the two are near-orthogonal, so each carries information the other does not.
        # A position is pruned only when it is uninformative OR untrustworthy.
        if teacher_top_k_log_probs is None or student_top_k_log_probs is None:
            raise ValueError("prune-opd metric ig_gated requires student and teacher top-k log-probs.")
        if overlap_mask is None:
            raise ValueError("prune-opd metric ig_gated requires overlap_mask.")

        overlap_mask = overlap_mask.to(device)
        overlap_ratio = overlap_mask.float().mean(dim=-1)
        info_gain = _local_info_gain(
            student_top_k_log_probs.to(device),
            teacher_top_k_log_probs.to(device),
            overlap_mask,
            restrict_to_overlap=bool(_cfg_get(config, "ig_restrict_to_overlap", True)),
        )
        gate = float(_cfg_get(config, "reliability_gate", 0.7))
        bad_event = ((info_gain < threshold) | (overlap_ratio < gate)) & valid_mask
        aux_tensors["prune_opd_info_gain"] = info_gain
        aux_tensors["prune_opd_overlap_ratio"] = overlap_ratio
    elif metric == "info_gain":
        # Reliability (overlap) answers "is the teacher familiar with this state".
        # It does not answer "does the teacher have anything left to teach here".
        # Offline measurement on 2.0M student-visited positions (R1-1.5B <- JustRL-1.5B)
        # shows the two are essentially uncorrelated (Pearson corr(O, IG) = -0.095), so an
        # overlap-driven weight carries almost no information about where the gradient is.
        # Here the drift event fires where the teacher adds little, i.e. small local KL.
        if teacher_top_k_log_probs is None:
            raise ValueError("prune-opd metric info_gain requires teacher_top_k_log_probs.")
        if student_top_k_log_probs is None:
            raise ValueError("prune-opd metric info_gain requires student_top_k_log_probs.")

        info_gain = _local_info_gain(
            student_top_k_log_probs.to(device),
            teacher_top_k_log_probs.to(device),
            overlap_mask.to(device) if overlap_mask is not None else None,
            restrict_to_overlap=bool(_cfg_get(config, "ig_restrict_to_overlap", True)),
        )
        # threshold is a KL in nats. Positions BELOW it are the uninformative ones.
        bad_event = (info_gain < threshold) & valid_mask
        aux_tensors["prune_opd_info_gain"] = info_gain
    else:
        raise NotImplementedError(f"Unsupported prune-opd metric: {metric}")

    cumulative_bad_event = bad_event.to(rm_scores.dtype).cumsum(dim=-1)
    if attenuate:
        weights = 1.0 - w_drop * cumulative_bad_event
        weights = torch.clamp(weights, min=0.0, max=1.0)
    else:
        # ICT truncates each trajectory AT its own boundary, so every retained position is
        # inside the allowance by construction and the monotone weight has nothing left to
        # suppress. Applying it anyway decays on a second, unrelated budget (1/w_drop) and
        # zeroes the tail of a prefix ICT paid to decode -- measured E/retained = 0.21 on
        # G2/G3/G4, i.e. 79% of the retained tokens were silently discarded.
        weights = torch.ones_like(cumulative_bad_event)
    weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
    loss_weights = weights + w_base
    loss_weights = torch.where(valid_mask, loss_weights, torch.zeros_like(loss_weights))

    # w_base is documented as a floor, but weights lives in [0, 1], so L = weights + w_base
    # actually spans [w_base, 1 + w_base] -- with the paper default w_base=0.5 that is
    # [0.5, 1.5], i.e. reliable positions get a 1.5x multiplier rather than a neutral 1.0.
    # Under token-mean aggregation the batch-mean of L therefore acts as an unannounced
    # learning-rate scale that drifts as the prune rate falls over training. Renormalizing
    # to mean 1 separates "redistribute weight across positions" from "scale the step size".
    if bool(_cfg_get(config, "normalize_loss_weights", False)):
        denom = (loss_weights * valid_mask).sum()
        count = valid_mask.sum()
        scale = torch.where(
            denom > positive_eps, count.to(loss_weights.dtype) / denom, torch.ones_like(denom)
        )
        loss_weights = loss_weights * scale
        loss_weights = torch.where(valid_mask, loss_weights, torch.zeros_like(loss_weights))
        # aux tensors travel through DataProto, which requires a common batch dim,
        # so log the scalar scale as one value per sequence rather than a bare scalar.
        aux_tensors["prune_opd_weight_scale"] = scale.reshape(1).expand(rm_scores.size(0)).clone()

    weighted_rm_scores = rm_scores * loss_weights.unsqueeze(-1)
    # E is the position count a reliability budget of 1/w_drop would have allowed, and it is
    # reported whether or not that budget is enforced: with attenuation off `weights` is
    # identically 1, so deriving E from it would silently turn E into the retained length and
    # blind exactly the diagnostic that caught the ICT weight mismatch. Compute it from the
    # bad-event count instead, so E and the retained length stay independently comparable.
    if w_drop > 0.0:
        within_budget = cumulative_bad_event < (1.0 / w_drop)
    else:
        within_budget = torch.ones_like(cumulative_bad_event, dtype=torch.bool)
    effective_response_length = (valid_mask & within_budget).sum(dim=-1).to(torch.float32)

    aux_tensors["prune_opd_weights"] = weights
    aux_tensors["prune_opd_loss_weights"] = loss_weights
    aux_tensors["prune_opd_bad_event"] = bad_event.to(torch.float32)
    aux_tensors["prune_opd_effective_response_length"] = effective_response_length
    return weighted_rm_scores, aux_tensors
