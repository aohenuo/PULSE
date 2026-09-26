"""Core tensor operations for SAE-based PULSE demonstration selection.

This module deliberately excludes datasets, model loading, baselines, metrics,
and experiment artifact handling. Callers provide precomputed SAE and SBERT
representations plus the discovery utilities needed to learn the PULSE weight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .configs import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig


def rank_candidate_sets(
    context_deltas: torch.Tensor,
    pulse_weight: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    """Eq. 7 signed scores and descending indices for complete contexts.

    context_deltas contains the SAE activation difference between each
    candidate prompt and its query-only prompt; shape is [candidates, features].
    """
    if context_deltas.ndim != 2 or pulse_weight.ndim != 1:
        raise ValueError("context_deltas must be 2D and pulse_weight must be 1D")
    if context_deltas.shape[1] != pulse_weight.numel():
        raise ValueError("context_deltas and pulse_weight must share feature dimension")
    if not torch.isfinite(context_deltas).all() or not torch.isfinite(pulse_weight).all():
        raise ValueError("ranking inputs must be finite")
    scores = context_deltas.float() @ pulse_weight.to(context_deltas.device).float()
    return scores, torch.argsort(scores, descending=True, stable=True).tolist()


@dataclass(frozen=True)
class PairFeatureStatistics:
    """Sufficient statistics over unordered within-query candidate pairs."""

    numerator: torch.Tensor
    delta_sum: torch.Tensor
    delta_square_sum: torch.Tensor
    pair_count: int


def pair_feature_statistics(
    utilities: torch.Tensor,
    activations: torch.Tensor,
) -> PairFeatureStatistics:
    """Compute all-pair Eq. 6 statistics without materializing pair deltas."""
    if utilities.ndim != 1:
        raise ValueError("utilities must be a 1D tensor")
    if activations.ndim != 2:
        raise ValueError("activations must be a 2D tensor")
    if activations.shape[0] != utilities.numel():
        raise ValueError("utilities and activations must have the same row count")
    if utilities.numel() < 2:
        raise ValueError("at least two candidates are required")

    values = utilities.to(dtype=torch.float64, device="cpu")
    acts = activations.to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(values).all() or not torch.isfinite(acts).all():
        raise ValueError("discovery inputs must be finite")
    n_candidates = int(values.numel())
    pair_count = n_candidates * (n_candidates - 1) // 2
    numerator = (
        n_candidates * (values.unsqueeze(1) * acts).sum(dim=0)
        - values.sum() * acts.sum(dim=0)
    )
    coefficients = torch.arange(
        n_candidates - 1,
        -n_candidates,
        -2,
        dtype=torch.float64,
    )
    delta_sum = coefficients @ acts
    activation_sum = acts.sum(dim=0)
    delta_square_sum = (
        n_candidates * acts.square().sum(dim=0) - activation_sum.square()
    ).clamp_min(0.0)
    return PairFeatureStatistics(
        numerator=numerator,
        delta_sum=delta_sum,
        delta_square_sum=delta_square_sum,
        pair_count=pair_count,
    )


def merge_pair_feature_statistics(
    statistics: Iterable[PairFeatureStatistics],
) -> PairFeatureStatistics:
    """Merge per-query statistics while preserving the global variance."""
    rows = list(statistics)
    if not rows:
        raise ValueError("statistics must be non-empty")
    shape = rows[0].numerator.shape
    device = rows[0].numerator.device
    dtype = rows[0].numerator.dtype
    for row in rows:
        tensors = (row.numerator, row.delta_sum, row.delta_square_sum)
        if any(tensor.shape != shape for tensor in tensors):
            raise ValueError("all statistics must have the same shape")
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("all statistics must use the same device")
        if any(tensor.dtype != dtype for tensor in tensors):
            raise ValueError("all statistics must use the same dtype")
    return PairFeatureStatistics(
        numerator=torch.stack([row.numerator for row in rows]).sum(dim=0),
        delta_sum=torch.stack([row.delta_sum for row in rows]).sum(dim=0),
        delta_square_sum=torch.stack(
            [row.delta_square_sum for row in rows]
        ).sum(dim=0),
        pair_count=sum(row.pair_count for row in rows),
    )


def feature_scores_from_statistics(
    statistics: PairFeatureStatistics,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Convert merged Eq. 6 sufficient statistics to dense feature scores."""
    if statistics.pair_count < 2:
        raise ValueError("at least two pairs are required")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    count = float(statistics.pair_count)
    centered_square_sum = (
        statistics.delta_square_sum
        - statistics.delta_sum.square() / count
    ).clamp_min(0.0)
    sample_variance = centered_square_sum / (count - 1.0)
    scores = (statistics.numerator / count) / torch.sqrt(
        sample_variance + float(epsilon)
    )
    return scores.to(dtype=torch.float32)


def sparse_signed_topk(
    dense_scores: torch.Tensor,
    topk_each_sign: int,
) -> torch.Tensor:
    """Keep deterministic top-k positive and negative PULSE features."""
    if dense_scores.ndim != 1:
        raise ValueError("dense_scores must be a 1D tensor")
    if not torch.isfinite(dense_scores).all():
        raise ValueError("dense_scores must be finite")
    if topk_each_sign < 0:
        raise ValueError("topk_each_sign must be non-negative")
    weights = torch.zeros_like(dense_scores)
    if topk_each_sign == 0:
        return weights
    for positive in (True, False):
        mask = dense_scores > 0 if positive else dense_scores < 0
        indices = mask.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            continue
        values = dense_scores[indices] if positive else -dense_scores[indices]
        order = torch.argsort(values, descending=True, stable=True)
        selected = indices[order[: min(topk_each_sign, int(indices.numel()))]]
        weights[selected] = dense_scores[selected]
    return weights


def learn_pulse_weight(
    discovery_utilities: Sequence[torch.Tensor],
    discovery_activations: Sequence[torch.Tensor],
    *,
    topk_each_sign: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Learn the sparse PULSE weight from independent discovery queries."""
    if len(discovery_utilities) != len(discovery_activations):
        raise ValueError("utility and activation query counts must match")
    if not discovery_utilities:
        raise ValueError("at least one discovery query is required")
    per_query = [
        pair_feature_statistics(utilities, activations)
        for utilities, activations in zip(
            discovery_utilities, discovery_activations, strict=True
        )
    ]
    dense_scores = feature_scores_from_statistics(
        merge_pair_feature_statistics(per_query), epsilon=epsilon
    )
    return sparse_signed_topk(dense_scores, topk_each_sign)


def standardize_scores(values: torch.Tensor) -> torch.Tensor:
    """Sample-standardize one relevance vector with a safe constant case."""
    if values.ndim != 1:
        raise ValueError("relevance scores must be one-dimensional")
    values = values.to(dtype=torch.float32)
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("relevance scores contain non-finite values")
    if values.numel() < 2:
        return torch.zeros_like(values)
    std = values.std(unbiased=True)
    if not torch.isfinite(std) or float(std.item()) <= 1e-8:
        return torch.zeros_like(values)
    return (values - values.mean()) / std


def pulse_eq9_relevance(
    query_sae: torch.Tensor,
    pool_sae: torch.Tensor,
    pulse_weight: torch.Tensor,
    *,
    beta: float = 0.3,
) -> torch.Tensor:
    """Compute literal Eq. 8-9 relevance using abs(weight) SAE geometry."""
    if query_sae.ndim != 1 or pool_sae.ndim != 2 or pulse_weight.ndim != 1:
        raise ValueError("query_sae and pulse_weight must be 1D; pool_sae must be 2D")
    if pool_sae.shape[1] != query_sae.numel() or pulse_weight.numel() != query_sae.numel():
        raise ValueError("all tensors must share the SAE feature dimension")
    if pool_sae.device != query_sae.device:
        raise ValueError("query_sae and pool_sae must use the same device")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    query = query_sae.to(dtype=torch.float32)
    pool = pool_sae.to(dtype=torch.float32)
    weight = pulse_weight.to(device=query.device, dtype=torch.float32)
    if not bool(torch.isfinite(query).all().item()):
        raise ValueError("query_sae contains non-finite values")
    if not bool(torch.isfinite(pool).all().item()):
        raise ValueError("pool_sae contains non-finite values")
    if not bool(torch.isfinite(weight).all().item()):
        raise ValueError("pulse_weight contains non-finite values")
    active = weight.nonzero(as_tuple=True)[0]
    if active.numel() == 0:
        raise ValueError("pulse_weight has no active features")
    magnitude = weight[active].abs()
    masked_query = query[active] * magnitude
    masked_pool = pool[:, active] * magnitude.unsqueeze(0)
    masked = F.cosine_similarity(masked_query.unsqueeze(0), masked_pool, dim=1)
    dense = F.cosine_similarity(query.unsqueeze(0), pool, dim=1)
    return (1.0 - beta) * standardize_scores(masked) + beta * standardize_scores(dense)


def semantic_relevance(
    query_sbert: torch.Tensor,
    pool_sbert: torch.Tensor,
) -> torch.Tensor:
    """Return normalized semantic cosine used only inside the fused method."""
    if query_sbert.ndim != 1 or pool_sbert.ndim != 2:
        raise ValueError("query_sbert must be 1D and pool_sbert must be 2D")
    if pool_sbert.shape[1] != query_sbert.numel():
        raise ValueError("query and pool SBERT dimensions must match")
    if pool_sbert.device != query_sbert.device:
        raise ValueError("query_sbert and pool_sbert must use the same device")
    query = F.normalize(query_sbert.to(dtype=torch.float32), dim=0)
    pool = F.normalize(pool_sbert.to(dtype=torch.float32), dim=1)
    return pool @ query


def classification_fusion_relevance(
    pulse_relevance: torch.Tensor,
    sbert_relevance: torch.Tensor,
    *,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Classification fusion extension with an outer row standardization."""
    if pulse_relevance.shape != sbert_relevance.shape:
        raise ValueError("PULSE and SBERT relevance must have the same shape")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    mixed = alpha * standardize_scores(pulse_relevance) + (
        1.0 - alpha
    ) * standardize_scores(sbert_relevance)
    return standardize_scores(mixed)


def generation_fusion_relevance(
    pulse_relevance: torch.Tensor,
    sbert_relevance: torch.Tensor,
    *,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Generation fusion extension restored to the literal Eq. 9 score scale."""
    if pulse_relevance.shape != sbert_relevance.shape:
        raise ValueError("PULSE and SBERT relevance must have the same shape")
    if pulse_relevance.ndim != 1:
        raise ValueError("fusion relevance must be one-dimensional")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    pulse = pulse_relevance.to(dtype=torch.float32)
    target_std = (
        pulse.std(unbiased=True)
        if pulse.numel() > 1
        else torch.zeros((), dtype=torch.float32, device=pulse.device)
    )
    if not torch.isfinite(target_std) or float(target_std.item()) <= 1e-8:
        return torch.zeros_like(pulse)
    mixed = alpha * standardize_scores(pulse) + (
        1.0 - alpha
    ) * standardize_scores(sbert_relevance)
    return standardize_scores(mixed) * target_std


def full_sae_greedy_select(
    relevance: torch.Tensor,
    pool_sae: torch.Tensor,
    *,
    n_shot: int,
    shortlist: int = 50,
    lambda_r: float = 0.3,
    pool_sae_normalized: torch.Tensor | None = None,
) -> list[int]:
    """Compose a context with top-L relevance and full-SAE redundancy."""
    if relevance.ndim != 1 or pool_sae.ndim != 2:
        raise ValueError("relevance must be 1D and pool_sae must be 2D")
    if pool_sae.shape[0] != relevance.numel():
        raise ValueError("pool_sae rows must match relevance")
    if pool_sae.device != relevance.device:
        raise ValueError("relevance and pool_sae must use the same device")
    if pool_sae_normalized is not None:
        if pool_sae_normalized.shape != pool_sae.shape:
            raise ValueError("pool_sae_normalized must match pool_sae")
        if pool_sae_normalized.device != pool_sae.device:
            raise ValueError("normalized and raw pool tensors must use the same device")
    if n_shot < 0 or shortlist < 0:
        raise ValueError("n_shot and shortlist must be non-negative")
    if lambda_r < 0:
        raise ValueError("lambda_r must be non-negative")
    if not torch.isfinite(relevance).all() or not torch.isfinite(pool_sae).all():
        raise ValueError("selection inputs must be finite")
    if n_shot == 0 or shortlist == 0 or relevance.numel() == 0:
        return []

    top_n = min(int(shortlist), int(relevance.numel()))
    top_global = torch.argsort(relevance, descending=True, stable=True)[:top_n]
    shortlist_scores = relevance[top_global].to(dtype=torch.float32)
    shortlist_sae = pool_sae[top_global].to(dtype=torch.float32)
    shortlist_geometry = (
        pool_sae_normalized[top_global].to(dtype=torch.float32)
        if pool_sae_normalized is not None
        else F.normalize(shortlist_sae, dim=1)
    )
    selected_local: list[int] = []
    selected_mask = torch.zeros(top_n, dtype=torch.bool, device=relevance.device)
    max_redundancy = torch.zeros(top_n, dtype=torch.float32, device=relevance.device)
    for _ in range(min(int(n_shot), top_n)):
        objective = shortlist_scores - float(lambda_r) * max_redundancy
        objective = objective.masked_fill(selected_mask, float("-inf"))
        pick = int(objective.argmax().item())
        selected_local.append(pick)
        selected_mask[pick] = True
        similarity = shortlist_geometry @ shortlist_geometry[pick]
        max_redundancy = (
            similarity
            if len(selected_local) == 1
            else torch.maximum(max_redundancy, similarity)
        )
    return [int(top_global[index].item()) for index in selected_local]


def _select_fused_context(
    query_sae: torch.Tensor,
    query_sbert: torch.Tensor,
    pool_sae: torch.Tensor,
    pool_sbert: torch.Tensor,
    pulse_weight: torch.Tensor,
    *,
    config: RetrievalConfig,
    generation: bool,
    pool_sae_normalized: torch.Tensor | None,
) -> list[int]:
    pulse = pulse_eq9_relevance(
        query_sae, pool_sae, pulse_weight, beta=config.beta
    )
    semantic = semantic_relevance(query_sbert, pool_sbert)
    if semantic.device != pulse.device:
        raise ValueError("SAE and SBERT representations must use the same device")
    fused = (
        generation_fusion_relevance(pulse, semantic, alpha=config.alpha)
        if generation
        else classification_fusion_relevance(pulse, semantic, alpha=config.alpha)
    )
    return full_sae_greedy_select(
        fused,
        pool_sae,
        n_shot=config.n_shot,
        shortlist=config.shortlist,
        lambda_r=config.lambda_r,
        pool_sae_normalized=pool_sae_normalized,
    )


def select_classification_context(
    query_sae: torch.Tensor,
    query_sbert: torch.Tensor,
    pool_sae: torch.Tensor,
    pool_sbert: torch.Tensor,
    pulse_weight: torch.Tensor,
    *,
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
    pool_sae_normalized: torch.Tensor | None = None,
) -> list[int]:
    """Select the formal alpha=0.5 classification context."""
    return _select_fused_context(
        query_sae,
        query_sbert,
        pool_sae,
        pool_sbert,
        pulse_weight,
        config=config,
        generation=False,
        pool_sae_normalized=pool_sae_normalized,
    )


def select_pulse_context(
    query_sae: torch.Tensor,
    pool_sae: torch.Tensor,
    pulse_weight: torch.Tensor,
    *,
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> list[int]:
    """Select by Eq. 8-9 relevance and full-SAE redundancy, without fusion.

    Returns zero-based pool indices in greedy selection order. The config's
    alpha is only used by the separately named semantic fusion selectors.
    """
    relevance = pulse_eq9_relevance(
        query_sae, pool_sae, pulse_weight, beta=config.beta
    )
    return full_sae_greedy_select(
        relevance, pool_sae, n_shot=config.n_shot,
        shortlist=config.shortlist, lambda_r=config.lambda_r,
    )


def select_generation_context(
    query_sae: torch.Tensor,
    query_sbert: torch.Tensor,
    pool_sae: torch.Tensor,
    pool_sbert: torch.Tensor,
    pulse_weight: torch.Tensor,
    *,
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
    pool_sae_normalized: torch.Tensor | None = None,
) -> list[int]:
    """Select the formal scale-matched alpha=0.5 generation context."""
    return _select_fused_context(
        query_sae,
        query_sbert,
        pool_sae,
        pool_sbert,
        pulse_weight,
        config=config,
        generation=True,
        pool_sae_normalized=pool_sae_normalized,
    )
