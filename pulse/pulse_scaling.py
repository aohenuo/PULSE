"""Shared helpers for the clean PULSE Layer A/B experiments.

This module intentionally keeps only the utilities used by the public
Layer A candidate-set and Layer B retrieval scripts.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .data import select_eval_examples
from .exemplar_utility import (
    _effective_topk_each_sign,
    _select_pair_indices,
    build_candidate_exemplars,
    candidate_controls,
    fit_residual_model,
    lexical_similarity,
    pair_signature,
    select_sparse_weight_vector,
)
from .metrics import (
    predict_label_and_confidence_gap_from_scores,
    score_candidate_label_sequences,
)
from .modeling import (
    DEFAULT_LAYER,
    TokenLengthCache,
    build_prompt_with_segments,
    capture_sae_acts_post_single,
    pooled_regions_from_prompt,
    region_spans_from_segments,
)


@dataclass
class ScalingConfig:
    mode: str = "S"
    model_name: str = "google/gemma-2-2b"
    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    layer: int | None = None
    seed: int = 42
    template_id: str = "default"

    disc_queries: int = 64
    candidate_pool_size: int = 32
    n_shot: int = 4
    disc_n_shot: int = 4
    eval_queries: int = 512
    max_pool_size: int = 2000

    calib_fraction: float = 0.15
    pool_fraction: float = 0.70
    n_calib_queries: int = 50

    topk_each_sign: int = 256
    max_topk_each_sign: int = 512
    k_positive: int = 96
    k_negative: int = 32
    min_utility_delta: float = 0.1
    use_magnitude_weighting: bool = True
    use_residualized: bool = True
    use_variance_norm: bool = True
    pooling: str = "mean"

    min_matched_pair_fraction: float = 0.5
    min_matched_pairs: int = 64
    max_pairs_per_query: int = 256
    min_effective_pairs_per_query: int = 64

    hf_cache_dir: str | None = None
    data_root: str | None = None


def _sample_prompt(text: str, template_id: str) -> str:
    if template_id in ("default", "qa"):
        return f"Input: {text}\nLabel:"
    if template_id == "block":
        return f"[Query]\nContent: {text}\nCategory:"
    return f"Input: {text}\nLabel:"


def _sae_acts_mean(model, sae, sae_acts_name: str, prompt: str) -> torch.Tensor:
    """Forward pass through model+SAE and mean-pool SAE activations."""
    acts = capture_sae_acts_post_single(model, sae, sae_acts_name, prompt)
    if acts.shape[0] > 1:
        return acts[1:].float().cpu().mean(dim=0)
    return acts[0].float().cpu()


def _zero_shot_prompt(instruction: str, query_text: str, template_id: str) -> str:
    prompt, _ = build_prompt_with_segments(instruction, [], query_text, template_id)
    return prompt


def _compute_static_vectors(
    model,
    sae,
    sae_acts_name: str,
    pool: List[Tuple[str, str]],
    template_id: str,
) -> torch.Tensor:
    """Compute one static SAE vector per pool exemplar."""
    vecs = []
    n = len(pool)
    for idx, (text, _label) in enumerate(pool):
        prompt = _sample_prompt(str(text), template_id)
        vecs.append(_sae_acts_mean(model, sae, sae_acts_name, prompt))
        if (idx + 1) % 200 == 0 or idx == 0:
            print(f"    static SAE vectors: {idx + 1}/{n}", flush=True)
    print(f"    static SAE vectors: {n}/{n} (done)", flush=True)
    return torch.stack(vecs)


def _split_data(
    train_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    cfg: ScalingConfig,
) -> tuple[
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    List[Tuple[str, str]],
]:
    """Split train/test rows into discovery donors, pool, and eval queries."""
    rng = random.Random(cfg.seed)
    shuffled = list(train_data)
    rng.shuffle(shuffled)

    n_calib = max(1, int(len(shuffled) * cfg.calib_fraction))
    n_pool = max(1, int(len(shuffled) * cfg.pool_fraction))

    calib_all = shuffled[:n_calib]
    pool = shuffled[n_calib : n_calib + n_pool]
    if cfg.max_pool_size > 0 and len(pool) > cfg.max_pool_size:
        pool = pool[: cfg.max_pool_size]

    rng2 = random.Random(cfg.seed + 1)
    calib_shuffled = list(calib_all)
    rng2.shuffle(calib_shuffled)
    n_dq = min(cfg.disc_queries, len(calib_shuffled) // 3)
    disc_queries_subset = calib_shuffled[:n_dq]
    disc_exemplars = calib_shuffled[n_dq:]

    n_mode_m = min(cfg.n_calib_queries, len(disc_exemplars) // 2)
    calib_for_m = disc_exemplars[:n_mode_m]
    disc_exemplars = disc_exemplars[n_mode_m:]

    eval_queries = select_eval_examples(test_data, seed=cfg.seed, max_queries=cfg.eval_queries)
    return calib_for_m, disc_queries_subset, disc_exemplars, pool, eval_queries


def _evaluate_accuracy(
    model,
    label_words: dict[str, str],
    instruction: str,
    exemplars: list[tuple[str, str, str]],
    query_text: str,
    gold_label: str,
    template_id: str,
) -> bool:
    prompt, _ = build_prompt_with_segments(instruction, exemplars, query_text, template_id)
    scores = score_candidate_label_sequences(model, prompt, label_words)
    pred, _ = predict_label_and_confidence_gap_from_scores(scores)
    return str(pred) == str(gold_label)


def _class_balanced_topk(scores: torch.Tensor, pool, label_words, n_shot: int):
    """Pick high-scoring examples while avoiding repeated labels where possible."""
    selected: list[int] = []
    label_count: dict[str, int] = {}
    for _ in range(min(n_shot, len(pool))):
        best_i, best_score = -1, float("-inf")
        for idx, score in enumerate(scores.tolist()):
            if idx in selected:
                continue
            label = str(pool[idx][1])
            adjusted = float(score) - 0.3 * label_count.get(label, 0)
            if adjusted > best_score:
                best_i, best_score = idx, adjusted
        if best_i < 0:
            break
        selected.append(best_i)
        label = str(pool[best_i][1])
        label_count[label] = label_count.get(label, 0) + 1
    return [
        (
            str(pool[i][0]),
            str(pool[i][1]),
            label_words.get(str(pool[i][1]), str(pool[i][1])),
        )
        for i in selected
    ]


def cosine_scores(query: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(query.unsqueeze(0), matrix)
