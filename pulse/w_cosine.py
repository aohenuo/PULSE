"""PULSE sparse feature discovery.

The clean release exposes a single discovery path used by both Layer A and
Layer B: learn a sparse SAE feature weight vector from paired candidate-set
utility differences.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .metrics import margin_for_gold_label_scores, score_candidate_label_sequences
from .modeling import TokenLengthCache
from .pulse_scaling import (
    _effective_topk_each_sign,
    _select_pair_indices,
    build_candidate_exemplars,
    build_prompt_with_segments,
    candidate_controls,
    fit_residual_model,
    lexical_similarity,
    pair_signature,
    pooled_regions_from_prompt,
    region_spans_from_segments,
    select_sparse_weight_vector,
)
from .stats import OnlineVectorStats


@dataclass
class WCosineConfig:
    disc_queries: int = 128
    candidate_pool_size: int = 32
    disc_n_shot: int = 4
    topk_each_sign: int = 256
    max_topk_each_sign: int = 512
    k_positive: int = 96
    k_negative: int = 32
    min_utility_delta: float = 0.1
    use_magnitude_weighting: bool = True
    use_residualized: bool = True
    use_variance_norm: bool = True
    min_matched_pair_fraction: float = 0.5
    min_matched_pairs: int = 64
    max_pairs_per_query: int = 256
    min_effective_pairs_per_query: int = 64
    pooling: str = "mean"
    seed: int = 42
    template_id: str = "default"


def discover_pulse_weight(
    *,
    model,
    sae,
    sae_acts_name: str,
    token_len: TokenLengthCache,
    disc_queries: List[Tuple[str, str]],
    disc_exemplars: List[Tuple[str, str]],
    label_words: Dict[str, str],
    instruction: str,
    cfg: WCosineConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Learn a sparse PULSE feature weight vector.

    Returns:
        `(pulse_weight, f_plus, f_minus, diagnostics)` where `pulse_weight` has
        shape `(d_sae,)` and nonzero entries only on selected PULSE features.
    """
    d_sae = int(sae.cfg.d_sae)
    numerator = torch.zeros(d_sae, dtype=torch.float32)
    stats = OnlineVectorStats.create(d_sae, device=torch.device("cpu"))
    pair_count = 0

    for query_idx, (query_text, query_gold_raw) in enumerate(disc_queries):
        query_gold = str(query_gold_raw)
        if query_gold not in label_words:
            continue

        zero_prompt, zero_segments = build_prompt_with_segments(
            instruction,
            [],
            query_text,
            cfg.template_id,
        )
        zero_spans, _ = region_spans_from_segments(zero_segments, token_len)
        zero_pooled = pooled_regions_from_prompt(
            model,
            sae,
            sae_acts_name,
            zero_prompt,
            zero_spans,
            pooling=cfg.pooling,
        )
        zero_flat = zero_pooled["all"]
        zero_scores = score_candidate_label_sequences(model, zero_prompt, label_words)
        zero_margin = margin_for_gold_label_scores(zero_scores, query_gold)

        rows = []
        for candidate_idx in range(cfg.candidate_pool_size):
            exemplars = build_candidate_exemplars(
                disc_exemplars,
                label_words,
                cfg.disc_n_shot,
                cfg.seed + query_idx * 10000 + candidate_idx * 997,
            )
            prompt, segments = build_prompt_with_segments(
                instruction,
                exemplars,
                query_text,
                cfg.template_id,
            )
            spans, prompt_len = region_spans_from_segments(segments, token_len)
            pooled = pooled_regions_from_prompt(
                model,
                sae,
                sae_acts_name,
                prompt,
                spans,
                pooling=cfg.pooling,
            )
            delta = pooled["all"] - zero_flat
            scores = score_candidate_label_sequences(model, prompt, label_words)
            margin = margin_for_gold_label_scores(scores, query_gold)
            avg_ex_len, label_token_total, entropy = candidate_controls(exemplars, token_len)

            rows.append(
                {
                    "query_index": query_idx,
                    "candidate_index": candidate_idx,
                    "prompt_len_tokens": prompt_len,
                    "avg_exemplar_len_tokens": avg_ex_len,
                    "label_token_total_len": label_token_total,
                    "shot_count": len(exemplars),
                    "class_balance_entropy": entropy,
                    "signature": pair_signature(
                        exemplars=exemplars,
                        prompt_len_tokens=prompt_len,
                        avg_exemplar_len_tokens=avg_ex_len,
                        label_token_total_len=label_token_total,
                        template_id=cfg.template_id,
                        length_bucket_width=256,
                        avg_exemplar_bucket_width=64,
                        label_token_bucket_width=32,
                    ),
                    "similarity_score": lexical_similarity(query_text, exemplars),
                    "utility": float(margin - zero_margin),
                    "zero_margin": float(zero_margin),
                    "candidate_margin": float(margin),
                    "delta_flat": delta.float(),
                }
            )

        residuals = (
            fit_residual_model(rows)
            if cfg.use_residualized
            else {idx: row["utility"] for idx, row in enumerate(rows)}
        )
        pair_indices, _diag = _select_pair_indices(
            rows,
            min_matched_pair_fraction=cfg.min_matched_pair_fraction,
            min_matched_pairs=cfg.min_matched_pairs,
        )
        if cfg.max_pairs_per_query > 0 and len(pair_indices) > cfg.max_pairs_per_query:
            pair_indices = random.Random(cfg.seed + query_idx * 1009).sample(
                pair_indices,
                k=cfg.max_pairs_per_query,
            )

        nonzero_pairs = []
        threshold_pairs = []
        for left_idx, right_idx in pair_indices:
            utility_delta = residuals[left_idx] - residuals[right_idx]
            if utility_delta == 0:
                continue
            nonzero_pairs.append((left_idx, right_idx, utility_delta))
            if abs(utility_delta) >= cfg.min_utility_delta:
                threshold_pairs.append((left_idx, right_idx, utility_delta))

        selected_pairs = (
            threshold_pairs
            if len(threshold_pairs) >= cfg.min_effective_pairs_per_query
            else nonzero_pairs
        )
        for left_idx, right_idx, utility_delta in selected_pairs:
            feature_delta = rows[left_idx]["delta_flat"] - rows[right_idx]["delta_flat"]
            weight = utility_delta if cfg.use_magnitude_weighting else math.copysign(1.0, utility_delta)
            numerator += weight * feature_delta
            stats.update(feature_delta)
            pair_count += 1

        if (query_idx + 1) % 16 == 0 or query_idx == 0:
            print(f"    disc {query_idx + 1}/{len(disc_queries)}: {len(selected_pairs)} pairs")

    eps = 1e-6
    if cfg.use_variance_norm:
        score_std = (numerator / max(pair_count, 1)) / torch.sqrt(stats.variance() + eps)
    else:
        score_std = numerator / max(pair_count, 1)

    effective_topk = _effective_topk_each_sign(
        cfg.topk_each_sign,
        pair_count,
        adaptive_topk_by_pairs=True,
        max_topk_each_sign=cfg.max_topk_each_sign,
    )
    full_weight = select_sparse_weight_vector(score_std, effective_topk)

    pos_mask = full_weight > 0
    neg_mask = full_weight < 0
    f_plus = torch.topk(
        full_weight.masked_fill(~pos_mask, float("-inf")),
        k=min(cfg.k_positive, int(pos_mask.sum().item())),
    ).indices
    f_minus = torch.topk(
        (-full_weight).masked_fill(~neg_mask, float("-inf")),
        k=min(cfg.k_negative, int(neg_mask.sum().item())),
    ).indices

    diag = {
        "pair_count": int(pair_count),
        "n_pos_features": int(f_plus.shape[0]),
        "n_neg_features": int(f_minus.shape[0]),
        "total_nonzero": int((full_weight != 0).sum().item()),
        "effective_topk_each_sign": int(effective_topk),
    }
    print(
        f"  PULSE Discovery: F+={diag['n_pos_features']}, "
        f"F-={diag['n_neg_features']}, pairs={pair_count}",
        flush=True,
    )
    return full_weight, f_plus, f_minus, diag

