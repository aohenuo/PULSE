"""Canonical PULSE scoring and retrieval.

Layer A scores candidate sets with the blend rule:

    blend_03 = 0.7 * z(W-cos) + 0.3 * z(raw SAE cosine)

Layer B retrieval defaults to the PULSE-only softD2+FPW policy: FPW computes
cosine after sqrt(|W|) feature weighting, then softD2 tilts pool-prior label
quotas toward the mode label among the top FPW neighbors.

The blend path remains available for explicit ablations and compatibility.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import torch
import torch.nn.functional as F


def _zscore(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std().clamp(min=1e-8)


def score_blend03(
    query_sae: torch.Tensor,
    pool_sae: torch.Tensor,
    active_features: torch.Tensor,
    pulse_weight: torch.Tensor,
    beta: float = 0.3,
    signed_w: bool = False,
) -> torch.Tensor:
    """Compute canonical PULSE blend scores for every pool item."""
    q_active = query_sae[active_features]
    pool_active = pool_sae[:, active_features]
    weights = pulse_weight[active_features]

    if signed_w:
        weighted_score = pool_active @ (q_active * weights)
    else:
        weights_abs = weights.abs()
        q_weighted = q_active * weights_abs
        pool_weighted = pool_active * weights_abs
        weighted_score = (
            F.normalize(pool_weighted, dim=1)
            @ F.normalize(q_weighted.unsqueeze(0), dim=1).T
        ).squeeze(1)

    base_cos = F.cosine_similarity(query_sae.unsqueeze(0), pool_sae)
    return (1.0 - beta) * _zscore(weighted_score) + beta * _zscore(base_cos)


def fpw_scores(
    query_sae: torch.Tensor,
    pool_sae: torch.Tensor,
    active_features: torch.Tensor,
    pulse_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute feature-prior weighted cosine scores using sqrt(|W|)."""
    if active_features.numel() == 0:
        return torch.zeros(pool_sae.shape[0], dtype=pool_sae.dtype, device=pool_sae.device)
    w_sqrt = pulse_weight[active_features].abs().sqrt()
    q_weighted = query_sae[active_features] * w_sqrt
    pool_weighted = pool_sae[:, active_features] * w_sqrt.unsqueeze(0)
    q_norm = F.normalize(q_weighted.unsqueeze(0), dim=1).squeeze(0)
    pool_norm = F.normalize(pool_weighted, dim=1)
    return pool_norm @ q_norm


def _largest_remainder(k: int, labels: Sequence[str], proportions: dict[str, float]) -> dict[str, int]:
    raw = {label: k * proportions.get(label, 0.0) for label in labels}
    quotas = {label: int(math.floor(value)) for label, value in raw.items()}
    remainders = {label: raw[label] - quotas[label] for label in labels}
    to_assign = int(k - sum(quotas.values()))
    order = sorted(labels, key=lambda label: (-remainders[label], label))
    for label in order[:to_assign]:
        quotas[label] += 1
    return quotas


def _label_bucket_pick(
    quotas: dict[str, int],
    pool_labels: Sequence[str],
    scores: torch.Tensor,
) -> list[int]:
    selected: list[int] = []
    target_count = int(sum(quotas.values()))
    labels = [str(label) for label in pool_labels]

    for label, count in quotas.items():
        if count <= 0:
            continue
        idxs = [idx for idx, pool_label in enumerate(labels) if pool_label == label]
        if not idxs:
            continue
        sub_scores = scores[idxs]
        top_local = sub_scores.topk(min(count, len(idxs))).indices.tolist()
        selected.extend(int(idxs[pos]) for pos in top_local)

    if len(selected) < target_count:
        mask = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
        for idx in selected:
            mask[idx] = False
        remaining = mask.nonzero(as_tuple=True)[0]
        if remaining.numel() > 0:
            fill_count = min(target_count - len(selected), int(remaining.numel()))
            top_remaining = scores[remaining].topk(fill_count).indices.tolist()
            selected.extend(int(remaining[pos].item()) for pos in top_remaining)

    return selected


def _pool_prior(pool_labels: Sequence[str]) -> tuple[list[str], dict[str, float]]:
    labels = [str(label) for label in pool_labels]
    counts = Counter(labels)
    total = max(len(labels), 1)
    ordered_labels = sorted(counts)
    return ordered_labels, {label: counts[label] / total for label in ordered_labels}


def _tilted_quota(base_quota: dict[str, int], mode_label: str, tilt: int) -> dict[str, int]:
    quota = dict(base_quota)
    if tilt <= 0 or mode_label not in quota:
        return quota
    donors = sorted(
        [label for label in quota if label != mode_label],
        key=lambda label: (-quota[label], label),
    )
    moved = 0
    while moved < tilt:
        progressed = False
        for donor in donors:
            if quota[donor] > 0 and moved < tilt:
                quota[donor] -= 1
                quota[mode_label] += 1
                moved += 1
                progressed = True
        if not progressed:
            break
    return quota


def select_softd2_fpw(
    scores: torch.Tensor,
    pool_labels: Sequence[str],
    n_shot: int,
    mode_topk: int = 8,
    tilt: int = 1,
) -> list[int]:
    """Select exemplars with softD2 label quotas and FPW scores."""
    if n_shot <= 0 or scores.numel() == 0:
        return []
    if int(scores.numel()) != len(pool_labels):
        raise ValueError("scores and pool_labels must have the same length")

    labels = [str(label) for label in pool_labels]
    k = min(int(n_shot), len(labels))
    all_labels, prior = _pool_prior(labels)
    quota = _largest_remainder(k, all_labels, prior)

    top_count = min(max(int(mode_topk), 0), int(scores.numel()))
    if tilt > 0 and top_count > 0:
        top_idx = scores.topk(top_count).indices.tolist()
        mode_label = Counter(labels[int(idx)] for idx in top_idx).most_common(1)[0][0]
        quota = _tilted_quota(quota, mode_label, int(tilt))

    return _label_bucket_pick(quota, labels, scores)


def select_retrieval_greedy(
    blend_scores: torch.Tensor,
    pool_sae: torch.Tensor,
    pool_labels: Sequence[str],
    active_features: torch.Tensor,
    n_shot: int,
    shortlist: int = 200,
    lambda_r: float = 0.3,
    lambda_b: float = 0.0,
) -> list[int]:
    """Select k pool indices by score with redundancy and optional label balance."""
    del pool_labels
    del lambda_b  # retained for CLI/backward compatibility; clean default is disabled.
    n_pool = int(blend_scores.shape[0])
    top_n = min(shortlist, n_pool)
    top_global = blend_scores.topk(top_n).indices.tolist()
    shortlist_scores = blend_scores[top_global]
    shortlist_sae = pool_sae[top_global][:, active_features]
    shortlist_sae_norm = F.normalize(shortlist_sae, dim=1)

    selected_local: list[int] = []
    for _ in range(min(n_shot, top_n)):
        best_idx = -1
        best_score = float("-inf")
        for local_idx in range(top_n):
            if local_idx in selected_local:
                continue
            score = float(shortlist_scores[local_idx].item())
            if selected_local:
                redundancy = float(
                    (shortlist_sae_norm[local_idx] @ shortlist_sae_norm[selected_local].T)
                    .max()
                    .item()
                )
                score -= lambda_r * redundancy
            if score > best_score:
                best_idx = local_idx
                best_score = score
        if best_idx < 0:
            break
        selected_local.append(best_idx)

    return [top_global[idx] for idx in selected_local]


def score_set(
    query_sae: torch.Tensor,
    set_sae: torch.Tensor,
    active_features: torch.Tensor,
    pulse_weight: torch.Tensor,
    beta: float = 0.3,
) -> float:
    """Score one candidate k-shot set by averaging canonical blend_03 scores."""
    scores = score_blend03(
        query_sae=query_sae,
        pool_sae=set_sae,
        active_features=active_features,
        pulse_weight=pulse_weight,
        beta=beta,
    )
    return float(scores.mean().item())
