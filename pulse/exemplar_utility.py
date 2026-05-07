"""Utility helpers for PULSE discovery."""
from __future__ import annotations

import json
import math
import random
import re
from typing import Dict, List, Sequence, Tuple

import torch

from .modeling import TokenLengthCache


def build_candidate_exemplars(
    train_examples: Sequence[Tuple[str, str]],
    label_words: Dict[str, str],
    n_shot_total: int,
    seed: int,
    sampling_mode: str = "random",
) -> List[Tuple[str, str, str]]:
    rng = random.Random(seed)
    target = min(n_shot_total, len(train_examples))
    if sampling_mode == "random":
        picked = rng.sample(list(train_examples), k=target)
    elif sampling_mode == "balanced":
        by_label: dict[str, list[tuple[str, str]]] = {}
        for text, label in train_examples:
            by_label.setdefault(str(label), []).append((str(text), str(label)))
        for rows in by_label.values():
            rng.shuffle(rows)
        label_order = list(by_label)
        rng.shuffle(label_order)
        picked = []
        while len(picked) < target:
            made_progress = False
            for label in label_order:
                rows = by_label.get(label, [])
                if rows:
                    picked.append(rows.pop())
                    made_progress = True
                    if len(picked) >= target:
                        break
            if not made_progress:
                break
        if len(picked) < target:
            remaining = [row for rows in by_label.values() for row in rows]
            rng.shuffle(remaining)
            picked.extend(remaining[: target - len(picked)])
    else:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

    return sorted(
        [
            (str(text), str(label), str(label_words.get(str(label), str(label))))
            for text, label in picked
        ],
        key=lambda item: (item[1], item[0]),
    )


def lexical_similarity(query_text: str, exemplars: Sequence[Tuple[str, str, str]]) -> float:
    query_tokens = set(re.findall(r"\w+", query_text.lower()))
    if not query_tokens:
        return 0.0
    sims = []
    for text, _gold, _label_word in exemplars:
        ex_tokens = set(re.findall(r"\w+", text.lower()))
        if ex_tokens:
            sims.append(len(query_tokens & ex_tokens) / max(len(query_tokens | ex_tokens), 1))
        else:
            sims.append(0.0)
    return float(sum(sims) / max(len(sims), 1))


def class_balance_entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [count / total for count in counts if count > 0]
    return float(-sum(prob * math.log(prob + 1e-12) for prob in probs))


def candidate_controls(
    exemplars: Sequence[Tuple[str, str, str]],
    token_len: TokenLengthCache,
) -> Tuple[float, int, float]:
    exemplar_lens = [max(token_len(text) - 1, 1) for text, _gold, _label_word in exemplars]
    avg_ex_len = float(sum(exemplar_lens) / max(len(exemplar_lens), 1))
    label_tok_total = int(
        sum(max(token_len(" " + label_word) - 1, 1) for _text, _gold, label_word in exemplars)
    )
    label_counts: dict[str, int] = {}
    for _text, gold_label, _label_word in exemplars:
        label_counts[gold_label] = label_counts.get(gold_label, 0) + 1
    return avg_ex_len, label_tok_total, class_balance_entropy(list(label_counts.values()))


def pair_signature(
    *,
    exemplars: Sequence[Tuple[str, str, str]],
    prompt_len_tokens: int,
    avg_exemplar_len_tokens: float,
    label_token_total_len: int,
    template_id: str,
    length_bucket_width: int,
    avg_exemplar_bucket_width: int,
    label_token_bucket_width: int,
) -> str:
    label_counts: dict[str, int] = {}
    for _text, gold_label, _label_word in exemplars:
        label_counts[gold_label] = label_counts.get(gold_label, 0) + 1
    payload = {
        "template_id": template_id,
        "shot": len(exemplars),
        "balance": sorted(label_counts.values(), reverse=True),
        "prompt_bucket": int(prompt_len_tokens // max(length_bucket_width, 1)),
        "avg_ex_bucket": int(avg_exemplar_len_tokens // max(avg_exemplar_bucket_width, 1)),
        "label_tok_bucket": int(label_token_total_len // max(label_token_bucket_width, 1)),
    }
    return json.dumps(payload, sort_keys=True)


def _signature_pair_indices(query_rows: Sequence[dict]) -> List[Tuple[int, int]]:
    by_sig: dict[str, list[int]] = {}
    for idx, row in enumerate(query_rows):
        by_sig.setdefault(row["signature"], []).append(idx)
    pairs = []
    for idxs in by_sig.values():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                pairs.append((idxs[i], idxs[j]))
    return pairs


def _all_pair_indices(size: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(size) for j in range(i + 1, size)]


def _select_pair_indices(
    query_rows: Sequence[dict],
    *,
    min_matched_pair_fraction: float,
    min_matched_pairs: int,
) -> Tuple[List[Tuple[int, int]], dict]:
    total_possible = (len(query_rows) * (len(query_rows) - 1)) // 2
    matched_pairs = _signature_pair_indices(query_rows)
    matched_count = len(matched_pairs)
    matched_fraction = float(matched_count / max(total_possible, 1))
    frac_threshold = min(max(float(min_matched_pair_fraction), 0.0), 1.0)
    min_pairs_threshold = max(int(min_matched_pairs), 1)
    use_fallback = (
        total_possible > 0
        and (matched_count < min_pairs_threshold or matched_fraction < frac_threshold)
    )
    selected_pairs = _all_pair_indices(len(query_rows)) if use_fallback else matched_pairs
    return selected_pairs, {
        "total_possible_pairs": int(total_possible),
        "matched_pair_count": int(matched_count),
        "selected_pair_count": int(len(selected_pairs)),
        "matched_pair_fraction": matched_fraction,
        "used_fallback": bool(use_fallback),
    }


def _effective_topk_each_sign(
    topk_each_sign: int,
    pair_count: int,
    *,
    adaptive_topk_by_pairs: bool,
    max_topk_each_sign: int,
) -> int:
    base = max(int(topk_each_sign), 1)
    if not adaptive_topk_by_pairs:
        return base
    cap = max(int(max_topk_each_sign), base)
    return int(min(cap, max(base, int(pair_count))))


def fit_residual_model(rows: Sequence[dict], loo: bool = True) -> Dict[int, float]:
    x, y = [], []
    for row in rows:
        x.append(
            [
                1.0,
                row["prompt_len_tokens"],
                row["avg_exemplar_len_tokens"],
                row["label_token_total_len"],
                row["shot_count"],
                row["class_balance_entropy"],
                row["zero_margin"],
            ]
        )
        y.append(row["utility"])
    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    beta = torch.linalg.lstsq(x_t, y_t.unsqueeze(1)).solution.squeeze(1)
    pred = x_t @ beta
    residual = y_t - pred
    if loo and len(rows) > x_t.shape[1] + 2:
        try:
            xtx_inv = torch.linalg.inv(x_t.T @ x_t)
            h_diag = (x_t @ xtx_inv * x_t).sum(dim=1)
            loo_residual = residual / (1.0 - h_diag).clamp(min=0.05)
            return {idx: float(loo_residual[idx].item()) for idx in range(len(rows))}
        except Exception:
            pass
    return {idx: float(residual[idx].item()) for idx in range(len(rows))}


def select_sparse_weight_vector(score_vector: torch.Tensor, topk_each_sign: int) -> torch.Tensor:
    weights = torch.zeros_like(score_vector)
    pos_mask = score_vector > 0
    neg_mask = score_vector < 0
    if pos_mask.any():
        idx = torch.topk(
            score_vector.masked_fill(~pos_mask, float("-inf")),
            k=min(topk_each_sign, int(pos_mask.sum().item())),
        ).indices
        weights[idx] = score_vector[idx]
    if neg_mask.any():
        idx = torch.topk(
            (-score_vector).masked_fill(~neg_mask, float("-inf")),
            k=min(topk_each_sign, int(neg_mask.sum().item())),
        ).indices
        weights[idx] = score_vector[idx]
    return weights


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[cursor][1]:
            end += 1
        avg = (cursor + end) / 2.0 + 1.0
        for pos in range(cursor, end + 1):
            ranks[indexed[pos][0]] = avg
        cursor = end + 1
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return float(cov / math.sqrt(vx * vy))


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(average_ranks(xs), average_ranks(ys))
