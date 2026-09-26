"""Five retained retrieval baselines, independent of the private research tree.

KNN-SAE and Lex-Sim preserve the matched SAE redundancy composer used by the
classification and CommonGen sidecars. CEIL denotes their static SBERT DPP
reconstruction, not the separately trained generation CEIL query projection.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .core import full_sae_greedy_select, standardize_scores

BASELINE_METHODS = ("sbert", "knn_sae", "ceil", "lex_sim", "random")
BASELINE_METADATA = {
    "sbert": {"variant": "pure SBERT cosine top-k; stable pool-order ties"},
    "knn_sae": {"variant": "sample-standardized full-SAE cosine with matched SAE redundancy composer"},
    "ceil": {
        "variant": "learning-free static SBERT conditional DPP reconstruction",
        "official_checkpoint": False,
        "exact_submitted_reproduction": False,
        "limitation": "Not the official trained CEIL checkpoint or the generation runner's learned query projection",
    },
    "lex_sim": {"variant": "lowercase whitespace Jaccard with matched SAE redundancy composer"},
    "random": {"variant": "uniform sampling without replacement", "seed_rule": "seed + 7919 * query_index"},
}


def lexical_token_set(text: str) -> set[str]:
    return set(text.lower().split())


def jaccard_similarity(query_tokens: set[str], document_tokens: set[str]) -> float:
    union = query_tokens | document_tokens
    return len(query_tokens & document_tokens) / len(union) if union else 0.0


def _pool(value: torch.Tensor | None, name: str, size: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor")
    if value.shape[0] != size or value.shape[1] == 0:
        raise ValueError(f"{name} must match pool_texts and have a nonempty feature dimension")
    value = value.detach().float()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")
    return value


def _query(value: torch.Tensor | None, name: str, pool: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.shape[0] != pool.shape[1]:
        raise ValueError(f"{name} must be 1D and match the pool feature dimension")
    if value.device != pool.device:
        raise ValueError(f"{name} and pool must use the same device")
    value = value.detach().float()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")
    return value


def static_sbert_dpp_kernel(query_vector: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
    query = F.normalize(query_vector.detach().float().reshape(1, -1), dim=1)
    candidates = F.normalize(candidate_vectors.detach().float(), dim=1)
    relevance = ((query @ candidates.T).squeeze(0) + 1.0).div(2.0).clamp_min(1e-6)
    similarity = ((candidates @ candidates.T) + 1.0).div(2.0)
    kernel = relevance[:, None] * similarity * relevance[None, :]
    kernel = (kernel + kernel.T).div(2.0)
    kernel.diagonal().add_(1e-6)
    return kernel


def greedy_map_dpp(kernel: torch.Tensor, k: int) -> list[int]:
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    selected: list[int] = []
    remaining = list(range(int(kernel.shape[0])))
    for _ in range(min(max(int(k), 0), len(remaining))):
        best_index, best_value = -1, float("-inf")
        for candidate in remaining:
            subset = selected + [candidate]
            sign, logabsdet = torch.linalg.slogdet(kernel[subset][:, subset])
            value = float(logabsdet.item()) if float(sign.item()) > 0 else float("-inf")
            if value > best_value:
                best_index, best_value = candidate, value
        if best_index < 0:
            break
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


def select_baseline(
    method: str, *, pool_texts: Sequence[str], query_text: str,
    n_shot: int = 4, seed: int = 42, query_index: int = 0,
    shortlist: int = 50, lambda_r: float = 0.3,
    query_sae: torch.Tensor | None = None, pool_sae: torch.Tensor | None = None,
    query_sbert: torch.Tensor | None = None, pool_sbert: torch.Tensor | None = None,
) -> list[int]:
    """Select pool indices in demonstration order using the declared variant."""
    if method not in BASELINE_METHODS:
        raise ValueError(f"unknown baseline: {method}")
    for name, value in (("n_shot", n_shot), ("shortlist", shortlist), ("query_index", query_index)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not math.isfinite(lambda_r) or lambda_r < 0:
        raise ValueError("lambda_r must be finite and non-negative")
    if isinstance(pool_texts, str) or not all(isinstance(text, str) for text in pool_texts):
        raise ValueError("pool_texts must be a sequence of strings")
    if not isinstance(query_text, str):
        raise ValueError("query_text must be a string")
    size = len(pool_texts)
    if method == "random":
        return random.Random(seed + 7919 * query_index).sample(range(size), k=min(n_shot, size))
    if method in ("sbert", "ceil"):
        pool = _pool(pool_sbert, "pool_sbert", size)
        query = _query(query_sbert, "query_sbert", pool)
        pool = F.normalize(pool, dim=1)
        query = F.normalize(query, dim=0)
        ranking = torch.argsort(pool @ query, descending=True, stable=True)
        if method == "sbert":
            return ranking[:n_shot].tolist()
        top = ranking[:shortlist]
        kernel = static_sbert_dpp_kernel(query, pool[top])
        return [int(top[index]) for index in greedy_map_dpp(kernel, n_shot)]
    pool = _pool(pool_sae, "pool_sae", size)
    if method == "knn_sae":
        query = _query(query_sae, "query_sae", pool)
        relevance = standardize_scores(F.normalize(pool, dim=1) @ F.normalize(query, dim=0))
    else:
        tokens = lexical_token_set(query_text)
        relevance = torch.tensor([
            jaccard_similarity(tokens, lexical_token_set(text)) for text in pool_texts
        ], dtype=torch.float32, device=pool.device)
    return full_sae_greedy_select(relevance, pool, n_shot=n_shot, shortlist=shortlist, lambda_r=lambda_r)
