#!/usr/bin/env python3
"""Small generative-task smoke test for PULSE exemplar selection.

This script intentionally leaves the existing classification pipeline untouched.
Supported tasks:

  - agnews_headline: Article body -> missing headline prefix
  - common_gen:     Concept set -> natural sentence

Utility is teacher-forced reference log-likelihood gain:

    U(S) = log p(target | prompt(query, S)) - log p(target | prompt(query, empty))

The smoke test checks whether a PULSE weight learned from this generative utility
can rank k-shot candidate sets or retrieve exemplars with higher true utility.
It is not meant to be a paper-scale benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import random as pyrandom
import re
import sys
import time
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pulse.data import load_agnews
from pulse.exemplar_utility import select_sparse_weight_vector, spearman
from pulse.modeling import (
    TokenLengthCache,
    load_model,
    load_sae_for_layer,
    pooled_regions_from_prompt,
    region_spans_from_segments,
)
from pulse.scoring import fpw_scores, score_blend03, select_retrieval_greedy, select_softd2_fpw
from pulse.stats import OnlineVectorStats


TASK_SPECS: dict[str, dict[str, str]] = {
    "agnews_headline": {
        "name": "agnews_headline_reconstruction",
        "instruction": "Generate the missing news headline from the article body.",
        "input_prefix": "Article",
        "output_prefix": "Headline",
    },
    "common_gen": {
        "name": "common_gen",
        "instruction": "Write one natural sentence that uses all of the given concepts.",
        "input_prefix": "Concepts",
        "output_prefix": "Sentence",
    },
    "gsm8k": {
        "name": "gsm8k",
        "instruction": "Solve the math word problem step by step. End with a line of the form '#### <number>'.",
        "input_prefix": "Question",
        "output_prefix": "Answer",
        "stop_on_newline": False,
    },
}


def _word_split(text: str) -> list[str]:
    return [part.strip() for part in str(text).replace("\n", " ").split() if part.strip()]


def make_headline_pairs(
    rows: Sequence[tuple[str, str]],
    *,
    target_words: int,
    max_body_words: int,
) -> list[tuple[str, str]]:
    """Convert AGNews text into (body, headline_prefix) generation pairs."""
    pairs: list[tuple[str, str]] = []
    min_words = target_words + 12
    for text, _label in rows:
        words = _word_split(text)
        if len(words) < min_words:
            continue
        target = " ".join(words[:target_words])
        body = " ".join(words[target_words : target_words + max_body_words])
        if body and target:
            pairs.append((body, target))
    return pairs


def _data_roots(data_root: str | None = None) -> list[Path]:
    roots: list[Path] = []
    if data_root:
        roots.append(Path(data_root).expanduser())
    roots.append(REPO_ROOT / "data")
    dedup: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            dedup.append(root)
    return dedup


def _format_concepts(raw: Any) -> str:
    if isinstance(raw, list):
        parts = [str(item).strip() for item in raw if str(item).strip()]
    else:
        parts = [
            item.strip()
            for item in re.split(r"[|,;]", str(raw))
            if item.strip()
        ]
    return ", ".join(parts)


def load_common_gen_pairs(split: str, data_root: str | None = None) -> list[tuple[str, str]]:
    candidates = [
        f"common_gen_{split}.jsonl",
        f"commongen_{split}.jsonl",
        f"common_gen_{split}.csv",
        f"commongen_{split}.csv",
    ]
    path = next(
        (root / name for root in _data_roots(data_root) for name in candidates if (root / name).exists()),
        None,
    )
    if path is None:
        raise FileNotFoundError(
            f"CommonGen split '{split}' not found. Expected one of {candidates} in data roots."
        )

    pairs: list[tuple[str, str]] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                source = _format_concepts(row.get("concepts") or row.get("source") or "")
                target = str(row.get("target") or row.get("sentence") or "").strip()
                if source and target:
                    pairs.append((source, target))
    else:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source = _format_concepts(row.get("concepts") or row.get("source") or "")
                target = str(row.get("target") or row.get("sentence") or "").strip()
                if source and target:
                    pairs.append((source, target))
    if not pairs:
        raise ValueError(f"Could not parse CommonGen file: {path}")
    return pairs


def load_gsm8k_pairs(split: str, data_root: str | None = None) -> list[tuple[str, str]]:
    name = {"train": "gsm8k_train.jsonl", "test": "gsm8k_test.jsonl"}.get(split)
    if name is None:
        raise ValueError(f"Unsupported gsm8k split: {split}")
    path = next((root / name for root in _data_roots(data_root) if (root / name).exists()), None)
    if path is None:
        raise FileNotFoundError(f"GSM8K split '{split}' not found. Expected {name} in data roots.")
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = str(row.get("question") or "").strip()
            a = str(row.get("answer") or "").strip()
            if q and a and "####" in a:
                pairs.append((q, a))
    return pairs


def load_task_pairs(
    task: str,
    *,
    target_words: int,
    max_body_words: int,
    data_root: str | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, str]]:
    if task == "agnews_headline":
        train_pairs = make_headline_pairs(
            load_agnews("train", data_root=data_root),
            target_words=target_words,
            max_body_words=max_body_words,
        )
        test_pairs = make_headline_pairs(
            load_agnews("test", data_root=data_root),
            target_words=target_words,
            max_body_words=max_body_words,
        )
        return train_pairs, test_pairs, TASK_SPECS[task]
    if task == "common_gen":
        return (
            load_common_gen_pairs("train", data_root=data_root),
            load_common_gen_pairs("validation", data_root=data_root),
            TASK_SPECS[task],
        )
    if task == "gsm8k":
        return (
            load_gsm8k_pairs("train", data_root=data_root),
            load_gsm8k_pairs("test", data_root=data_root),
            TASK_SPECS[task],
        )
    raise ValueError(f"Unsupported generative task: {task}")


def build_gen_prompt_with_segments(
    exemplars: Sequence[tuple[str, str]],
    query_body: str,
    spec: dict[str, str],
) -> tuple[str, list[tuple[str, str]]]:
    segments: list[tuple[str, str]] = [("none", spec["instruction"] + "\n\n")]
    for body, target in exemplars:
        segments.extend(
            [
                ("none", f"{spec['input_prefix']}: "),
                ("ex_input", body),
                ("none", f"\n{spec['output_prefix']}: "),
                ("ex_label", target),
                ("none", "\n\n"),
            ]
        )
    segments.extend(
        [
            ("none", f"{spec['input_prefix']}: "),
            ("query", query_body),
            ("none", f"\n{spec['output_prefix']}:"),
        ]
    )
    return "".join(text for _region, text in segments), segments


def build_single_exemplar_prompt(
    body: str,
    target: str,
    spec: dict[str, str],
) -> tuple[str, list[tuple[str, str]]]:
    segments = [
        ("none", f"{spec['input_prefix']}: "),
        ("ex_input", body),
        ("none", f"\n{spec['output_prefix']}: "),
        ("ex_label", target),
    ]
    return "".join(text for _region, text in segments), segments


def target_logprob(model, prompt: str, target: str) -> tuple[float, int]:
    """Teacher-forced sum log p(target | prompt), with target token count."""
    tok = model.tokenizer
    device = next(model.parameters()).device
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    target_ids = tok.encode(" " + target, add_special_tokens=False)
    if not target_ids:
        return float("-inf"), 0

    full_ids = prompt_ids + target_ids
    tokens = torch.tensor([full_ids], dtype=torch.long, device=device)
    answer_tokens = torch.tensor(target_ids, dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    answer_len = len(target_ids)

    with torch.no_grad():
        logits = model(tokens)
    pred_logits = logits[0, prompt_len - 1 : prompt_len - 1 + answer_len, :].float()
    logprobs = F.log_softmax(pred_logits, dim=-1)
    lp = logprobs.gather(-1, answer_tokens.unsqueeze(-1)).squeeze(-1).sum()
    return float(lp.item()), answer_len


def pooled_all_vector(model, sae, sae_acts_name: str, token_len: TokenLengthCache, prompt: str, segments):
    spans, _total = region_spans_from_segments(segments, token_len)
    pooled = pooled_regions_from_prompt(model, sae, sae_acts_name, prompt, spans, pooling="mean")
    return pooled["all"].float().cpu()


def sample_exemplars(pool: Sequence[tuple[str, str]], n_shot: int, seed: int) -> list[tuple[str, str]]:
    rng = pyrandom.Random(seed)
    return list(rng.sample(list(pool), k=min(n_shot, len(pool))))


def _gen_utility_score(
    model,
    prompt: str,
    target_text: str,
    *,
    spec: dict[str, Any],
    max_new_tokens: int,
    task_name: str,
) -> float:
    pred = generate_completion(model, prompt, max_new_tokens=max_new_tokens, spec=spec)
    if task_name == "gsm8k":
        pred_ans = extract_gsm8k_answer(pred)
        gold_ans = extract_gsm8k_answer(target_text)
        return 1.0 if (pred_ans is not None and pred_ans == gold_ans) else 0.0
    scores = score_generation(task_name, "", pred, target_text)
    return float(scores.get("score", 0.0))


def discover_generative_pulse(
    *,
    model,
    sae,
    sae_acts_name: str,
    token_len: TokenLengthCache,
    disc_queries: Sequence[tuple[str, str]],
    disc_exemplars: Sequence[tuple[str, str]],
    spec: dict[str, str],
    n_shot: int,
    candidate_pool_size: int,
    topk_each_sign: int,
    min_utility_delta: float,
    seed: int,
    utility_mode: str = "logprob",
    gen_max_new_tokens: int = 200,
) -> tuple[torch.Tensor, dict[str, Any]]:
    d_sae = int(sae.cfg.d_sae)
    numerator = torch.zeros(d_sae, dtype=torch.float32)
    stats = OnlineVectorStats.create(d_sae, device=torch.device("cpu"))
    pair_count = 0
    task_name = spec.get("name", "")

    for qi, (query_body, target) in enumerate(disc_queries):
        z_prompt, z_segments = build_gen_prompt_with_segments([], query_body, spec)
        z_vec = pooled_all_vector(model, sae, sae_acts_name, token_len, z_prompt, z_segments)
        if utility_mode == "sampled_em":
            z_score = _gen_utility_score(
                model, z_prompt, target, spec=spec,
                max_new_tokens=gen_max_new_tokens, task_name=task_name,
            )
        else:
            z_lp, z_len = target_logprob(model, z_prompt, target)
            if z_len <= 0:
                continue

        rows: list[dict[str, Any]] = []
        for ci in range(candidate_pool_size):
            exs = sample_exemplars(disc_exemplars, n_shot, seed + qi * 10000 + ci * 997)
            prompt, segments = build_gen_prompt_with_segments(exs, query_body, spec)
            cand_vec = pooled_all_vector(model, sae, sae_acts_name, token_len, prompt, segments)
            if utility_mode == "sampled_em":
                cand_score = _gen_utility_score(
                    model, prompt, target, spec=spec,
                    max_new_tokens=gen_max_new_tokens, task_name=task_name,
                )
                util_value = cand_score - z_score
            else:
                cand_lp, _ = target_logprob(model, prompt, target)
                util_value = cand_lp - z_lp
            rows.append(
                {
                    "delta_flat": cand_vec - z_vec,
                    "utility": util_value,
                    "candidate_index": ci,
                }
            )

        pair_deltas: list[tuple[int, int, float]] = []
        for li, ri in combinations(range(len(rows)), 2):
            dt = float(rows[li]["utility"] - rows[ri]["utility"])
            if dt != 0.0:
                pair_deltas.append((li, ri, dt))
        selected = [
            (li, ri, dt)
            for li, ri, dt in pair_deltas
            if abs(dt) >= min_utility_delta
        ] or pair_deltas

        for li, ri, dt in selected:
            df = rows[li]["delta_flat"] - rows[ri]["delta_flat"]
            numerator += float(dt) * df.float()
            stats.update(df.float())
            pair_count += 1

        print(
            f"    disc {qi + 1}/{len(disc_queries)}: "
            f"{len(selected)} pairs, utility range="
            f"{min(r['utility'] for r in rows):+.3f}..{max(r['utility'] for r in rows):+.3f}",
            flush=True,
        )

    if pair_count == 0:
        raise RuntimeError("No nonzero utility pairs found; increase candidate_pool_size.")

    eps = 1e-6
    score_std = (numerator / pair_count) / torch.sqrt(stats.variance() + eps)
    pulse_weight = select_sparse_weight_vector(score_std, topk_each_sign)
    diag = {
        "pair_count": int(pair_count),
        "nnz": int((pulse_weight != 0).sum().item()),
        "topk_each_sign": int(topk_each_sign),
    }
    return pulse_weight, diag


def compute_pool_vectors(
    *,
    model,
    sae,
    sae_acts_name: str,
    token_len: TokenLengthCache,
    pool: Sequence[tuple[str, str]],
    spec: dict[str, str],
) -> torch.Tensor:
    vecs = []
    for i, (body, target) in enumerate(pool):
        prompt, segments = build_single_exemplar_prompt(body, target, spec)
        vecs.append(pooled_all_vector(model, sae, sae_acts_name, token_len, prompt, segments))
        if (i + 1) % 32 == 0:
            print(f"    pool vec {i + 1}/{len(pool)}", flush=True)
    return torch.stack(vecs, dim=0)


def set_utility(
    model,
    query_body: str,
    target: str,
    exemplars: Sequence[tuple[str, str]],
    spec: dict[str, str],
) -> float:
    z_prompt, _ = build_gen_prompt_with_segments([], query_body, spec)
    prompt, _ = build_gen_prompt_with_segments(exemplars, query_body, spec)
    z_lp, _ = target_logprob(model, z_prompt, target)
    cand_lp, _ = target_logprob(model, prompt, target)
    return float(cand_lp - z_lp)


def _norm_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower())


def token_f1(prediction: str, reference: str) -> float:
    pred = _norm_tokens(prediction)
    ref = _norm_tokens(reference)
    if not pred or not ref:
        return 0.0
    ref_counts: dict[str, int] = {}
    for tok in ref:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    overlap = 0
    for tok in pred:
        count = ref_counts.get(tok, 0)
        if count > 0:
            overlap += 1
            ref_counts[tok] = count - 1
    if overlap <= 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return float(2 * precision * recall / (precision + recall))


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = _norm_tokens(prediction)
    ref = _norm_tokens(reference)
    if not pred or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for pred_tok in pred:
        cur = [0] * (len(ref) + 1)
        for j, ref_tok in enumerate(ref, start=1):
            if pred_tok == ref_tok:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if lcs <= 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return float(2 * precision * recall / (precision + recall))


def concept_coverage(source: str, prediction: str) -> float:
    concepts = [concept.strip().lower() for concept in str(source).split(",") if concept.strip()]
    if not concepts:
        return 0.0
    pred_text = " ".join(_norm_tokens(prediction))
    hits = 0
    for concept in concepts:
        concept_text = " ".join(_norm_tokens(concept))
        if concept_text and concept_text in pred_text:
            hits += 1
    return float(hits / len(concepts))


_GSM8K_FINAL_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
_GSM8K_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_gsm8k_answer(text: str) -> str | None:
    if not text:
        return None
    match = _GSM8K_FINAL_RE.search(text)
    if match is not None:
        raw = match.group(1)
    else:
        nums = _GSM8K_NUM_RE.findall(text)
        if not nums:
            return None
        raw = nums[-1]
    raw = raw.replace(",", "").rstrip(".")
    try:
        value = float(raw)
    except ValueError:
        return None
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def score_generation(task_name: str, source: str, prediction: str, reference: str) -> dict[str, float]:
    f1 = token_f1(prediction, reference)
    rouge_l = rouge_l_f1(prediction, reference)
    if task_name == "gsm8k":
        pred_ans = extract_gsm8k_answer(prediction)
        ref_ans = extract_gsm8k_answer(reference)
        em = 1.0 if (pred_ans is not None and pred_ans == ref_ans) else 0.0
        out = {
            "score": em,
            "exact_match": em,
            "token_f1": f1,
            "rouge_l": rouge_l,
            "has_final_marker": 1.0 if "####" in (prediction or "") else 0.0,
        }
        return {key: float(value) for key, value in out.items()}
    coverage = concept_coverage(source, prediction) if task_name == "common_gen" else None
    if coverage is None:
        overall = 0.5 * f1 + 0.5 * rouge_l
        out = {"score": overall, "token_f1": f1, "rouge_l": rouge_l}
    else:
        overall = 0.5 * coverage + 0.25 * f1 + 0.25 * rouge_l
        out = {
            "score": overall,
            "concept_coverage": coverage,
            "token_f1": f1,
            "rouge_l": rouge_l,
        }
    return {key: float(value) for key, value in out.items()}


def clean_completion_text(text: str, spec: dict[str, Any] | None = None) -> str:
    if spec is None or spec.get("stop_on_newline", True):
        text = text.split("\n", 1)[0]
        markers = ("Concepts:", "Sentence:", "Article:", "Headline:")
    else:
        markers = (
            f"\n{spec['input_prefix']}:",
            f"\n\n{spec['input_prefix']}:",
        )
    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def generate_completion(model, prompt: str, max_new_tokens: int, spec: dict[str, Any] | None = None) -> str:
    device = next(model.parameters()).device
    tokens = model.to_tokens(prompt).to(device)
    with torch.no_grad():
        output_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            stop_at_eos=True,
            do_sample=False,
            use_past_kv_cache=True,
            return_type="tokens",
            verbose=False,
        )
    generated = output_tokens[0, tokens.shape[1] :].detach().cpu().tolist()
    text = model.tokenizer.decode(generated, skip_special_tokens=True)
    return clean_completion_text(text, spec=spec)


def generate_completions(
    model,
    prompts: Sequence[str],
    max_new_tokens: int,
    spec: dict[str, Any] | None = None,
    gen_batch_size: int = 0,
) -> list[str]:
    if not prompts:
        return []
    chunk = gen_batch_size if gen_batch_size and gen_batch_size > 0 else len(prompts)
    completions: list[str] = []
    for start in range(0, len(prompts), chunk):
        sub = list(prompts[start : start + chunk])
        if len(sub) == 1:
            completions.append(
                generate_completion(model, sub[0], max_new_tokens=max_new_tokens, spec=spec)
            )
            continue
        device = next(model.parameters()).device
        tokens = model.to_tokens(sub, padding_side="left").to(device)
        with torch.no_grad():
            output_tokens = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                stop_at_eos=True,
                do_sample=False,
                use_past_kv_cache=True,
                return_type="tokens",
                verbose=False,
            )
        generated_batch = output_tokens[:, tokens.shape[1] :].detach().cpu().tolist()
        for generated in generated_batch:
            text = model.tokenizer.decode(generated, skip_special_tokens=True)
            completions.append(clean_completion_text(text, spec=spec))
    return completions


def downstream_eval_set(
    *,
    model,
    source: str,
    target: str,
    exemplars: Sequence[tuple[str, str]],
    spec: dict[str, str],
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt, _segments = build_gen_prompt_with_segments(exemplars, source, spec)
    prediction = generate_completion(model, prompt, max_new_tokens=max_new_tokens, spec=spec)
    scores = score_generation(spec["name"], source, prediction, target)
    return {
        "prediction": prediction,
        "target": target,
        "scores": scores,
    }


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def mean_score_dict(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: mean([row[key] for row in rows if key in row]) for key in keys}


def order_indices_for_prompt(
    idx_list: Sequence[int],
    *,
    mode: str,
    scores: torch.Tensor,
    pool_vectors: torch.Tensor,
    query_vector: torch.Tensor,
) -> list[int]:
    """Order retrieved exemplars before building the prompt.

    The classification ordering rules used label structure. Generative tasks do
    not have pool labels, so keep only label-free orderings.
    """
    ordered = [int(idx) for idx in idx_list]
    if mode == "as_selected" or len(ordered) <= 1:
        return ordered
    if mode == "score_desc":
        return sorted(ordered, key=lambda idx: -float(scores[idx].item()))
    if mode == "score_asc":
        return sorted(ordered, key=lambda idx: float(scores[idx].item()))
    if mode != "semantic_flow":
        raise ValueError(f"Unsupported prompt_order: {mode}")

    vecs = F.normalize(pool_vectors[torch.tensor(ordered, dtype=torch.long)].float(), dim=1)
    q = F.normalize(query_vector.float().unsqueeze(0), dim=1).squeeze(0)
    pairwise = vecs @ vecs.T
    q_sims = vecs @ q

    if len(ordered) <= 8:
        best_perm: tuple[int, ...] | None = None
        best_score = float("-inf")
        for perm in permutations(range(len(ordered))):
            smooth = 0.0
            for left, right in zip(perm[:-1], perm[1:]):
                smooth += float(pairwise[left, right].item())
            smooth += float(q_sims[perm[-1]].item())
            if smooth > best_score:
                best_score = smooth
                best_perm = perm
        assert best_perm is not None
        return [ordered[pos] for pos in best_perm]

    remaining = set(range(len(ordered)))
    start = min(remaining, key=lambda pos: float(q_sims[pos].item()))
    path = [start]
    remaining.remove(start)
    while remaining:
        last = path[-1]
        nxt = max(remaining, key=lambda pos: float(pairwise[last, pos].item()))
        path.append(nxt)
        remaining.remove(nxt)
    return [ordered[pos] for pos in path]


def _zscore_tensor(t: torch.Tensor) -> torch.Tensor:
    return (t - t.mean()) / t.std().clamp(min=1e-8)


def mask_to_score_shortlist(
    selection_scores: torch.Tensor,
    shortlist_scores: torch.Tensor,
    shortlist: int,
) -> torch.Tensor:
    """Keep selection scores only inside a shortlist defined by another score."""
    s = min(int(shortlist), int(selection_scores.numel()))
    if s >= int(selection_scores.numel()):
        return selection_scores
    keep = shortlist_scores.topk(s).indices
    masked = torch.full_like(selection_scores, float("-inf"))
    masked[keep] = selection_scores[keep]
    return masked


def score_blend_variant(
    query_sae: torch.Tensor,
    pool_sae: torch.Tensor,
    active_features: torch.Tensor,
    pulse_weight: torch.Tensor,
    *,
    beta: float,
    signed_w: bool,
    raw_scores: torch.Tensor | None,
) -> torch.Tensor:
    if raw_scores is None:
        return score_blend03(
            query_sae,
            pool_sae,
            active_features,
            pulse_weight,
            beta=beta,
            signed_w=signed_w,
        )

    q_active = query_sae[active_features]
    s_active = pool_sae[:, active_features]
    w = pulse_weight[active_features]
    if signed_w:
        wc = s_active @ (q_active * w)
    else:
        w_abs = w.abs()
        q_w = q_active * w_abs
        s_w = s_active * w_abs
        wc = (F.normalize(s_w, dim=1) @ F.normalize(q_w.unsqueeze(0), dim=1).T).squeeze(1)
    return (1.0 - beta) * _zscore_tensor(wc) + beta * _zscore_tensor(raw_scores.float())


def evaluate(
    *,
    model,
    sae,
    sae_acts_name: str,
    token_len: TokenLengthCache,
    pulse_weight: torch.Tensor,
    pool: Sequence[tuple[str, str]],
    pool_vectors: torch.Tensor,
    eval_queries: Sequence[tuple[str, str]],
    spec: dict[str, str],
    n_shot: int,
    n_candidates: int,
    beta: float,
    signed_w: bool,
    retrieval_score: str,
    retrieval_shortlist: int,
    softd2_mode_topk: int,
    softd2_tilt: int,
    lambda_r: float,
    prompt_order: str,
    seed: int,
    max_new_tokens: int,
    run_downstream: bool,
    gen_batch_size: int = 0,
) -> dict[str, Any]:
    active = pulse_weight.nonzero(as_tuple=True)[0]
    if active.numel() == 0:
        raise RuntimeError("Learned PULSE weight has no active features.")
    pool_labels = ["gen"] * len(pool)

    candidate_rows: list[dict[str, float]] = []
    retrieval = {"pulse": []}
    downstream: dict[str, dict[str, list[dict[str, float]]]] = {
        "candidate_set": {},
        "retrieval": {},
    }
    generation_rows: list[dict[str, Any]] = []

    for qi, (query_body, target) in enumerate(eval_queries):
        z_prompt, z_segments = build_gen_prompt_with_segments([], query_body, spec)
        q_vec = pooled_all_vector(model, sae, sae_acts_name, token_len, z_prompt, z_segments)
        blend_scores = score_blend_variant(
            q_vec,
            pool_vectors,
            active,
            pulse_weight,
            beta=beta,
            signed_w=signed_w,
            raw_scores=None,
        )
        fpw_pool_scores = fpw_scores(q_vec, pool_vectors, active, pulse_weight)
        if retrieval_score == "fpw":
            pulse_pool_scores = (
                0.5 * _zscore_tensor(fpw_pool_scores.float())
                + 0.5 * _zscore_tensor(blend_scores.float())
                if signed_w
                else fpw_pool_scores
            )
        elif retrieval_score == "softd2_fpw":
            pulse_pool_scores = fpw_pool_scores
        else:
            pulse_pool_scores = blend_scores

        rng = pyrandom.Random(seed + 500000 + qi * 10000)
        candidate_sets = [
            rng.sample(range(len(pool)), k=min(n_shot, len(pool)))
            for _ in range(n_candidates)
        ]
        true_utils = [
            set_utility(model, query_body, target, [pool[i] for i in idxs], spec)
            for idxs in candidate_sets
        ]
        pulse_set_scores = [
            float(pulse_pool_scores[torch.tensor(idxs, dtype=torch.long)].mean().item())
            for idxs in candidate_sets
        ]
        pulse_set_i = max(range(n_candidates), key=lambda i: pulse_set_scores[i])

        row = {
            "spearman_pulse": spearman(pulse_set_scores, true_utils),
            "pulse_selected_utility": true_utils[pulse_set_i],
        }
        candidate_rows.append(row)

        shortlist_size = min(retrieval_shortlist, len(pool))
        if retrieval_score == "softd2_fpw":
            pulse_idx = select_softd2_fpw(
                scores=fpw_pool_scores,
                pool_labels=pool_labels,
                n_shot=n_shot,
                mode_topk=softd2_mode_topk,
                tilt=softd2_tilt,
            )
        else:
            retrieval_blend_scores = pulse_pool_scores
            if retrieval_score == "fpw" and signed_w:
                retrieval_blend_scores = mask_to_score_shortlist(
                    selection_scores=pulse_pool_scores,
                    shortlist_scores=fpw_pool_scores,
                    shortlist=shortlist_size,
                )
            pulse_idx = select_retrieval_greedy(
                blend_scores=retrieval_blend_scores,
                pool_sae=pool_vectors,
                pool_labels=pool_labels,
                active_features=active,
                n_shot=n_shot,
                shortlist=shortlist_size,
                lambda_r=lambda_r,
                lambda_b=0.0,
            )
        pulse_prompt_idx = order_indices_for_prompt(
            pulse_idx,
            mode=prompt_order,
            scores=pulse_pool_scores,
            pool_vectors=pool_vectors,
            query_vector=q_vec,
        )
        retrieval["pulse"].append(
            set_utility(model, query_body, target, [pool[i] for i in pulse_prompt_idx], spec)
        )

        if run_downstream:
            method_sets = {
                ("candidate_set", "pulse"): candidate_sets[pulse_set_i],
                ("retrieval", "pulse"): pulse_prompt_idx,
            }
            q_generation: dict[str, Any] = {
                "query_index": qi,
                "source": query_body,
                "target": target,
                "candidate_set": {},
                "retrieval": {},
            }
            downstream_inputs: list[tuple[str, str, str]] = []
            for (scope, method), idxs in method_sets.items():
                prompt, _segments = build_gen_prompt_with_segments(
                    [pool[i] for i in idxs],
                    query_body,
                    spec,
                )
                downstream_inputs.append((scope, method, prompt))
            predictions = generate_completions(
                model,
                [prompt for _scope, _method, prompt in downstream_inputs],
                max_new_tokens=max_new_tokens,
                spec=spec,
                gen_batch_size=gen_batch_size,
            )
            for (scope, method, _prompt), prediction in zip(downstream_inputs, predictions):
                scores = score_generation(spec["name"], query_body, prediction, target)
                eval_row = {
                    "prediction": prediction,
                    "target": target,
                    "scores": scores,
                }
                downstream[scope].setdefault(method, []).append(eval_row["scores"])
                q_generation[scope][method] = eval_row
            generation_rows.append(q_generation)
        print(
            f"    eval {qi + 1}/{len(eval_queries)}: "
            f"retrieval pulse={retrieval['pulse'][-1]:+.3f} "
            + (
                f" downstream={downstream['retrieval']['pulse'][-1]['score']:.3f}"
                if run_downstream and downstream["retrieval"].get("pulse")
                else ""
            ),
            flush=True,
        )
        if run_downstream:
            em_parts = []
            for scope_name in ("retrieval", "candidate_set"):
                for method_name, rows in downstream[scope_name].items():
                    if not rows:
                        continue
                    last = rows[-1]
                    em_val = last.get("exact_match", last.get("score"))
                    if em_val is None:
                        continue
                    em_parts.append(f"{scope_name[0]}/{method_name}={em_val:.0f}")
            if em_parts:
                print(f"      EM[q{qi+1}]: " + " ".join(em_parts), flush=True)

    return {
        "candidate_set": {
            key: mean([row[key] for row in candidate_rows if key in row])
            for key in sorted({key for row in candidate_rows for key in row})
        },
        "retrieval": {name: mean(vals) for name, vals in retrieval.items()},
        "downstream": {
            scope: {
                method: mean_score_dict(rows)
                for method, rows in methods.items()
            }
            for scope, methods in downstream.items()
        },
        "generations": generation_rows,
        "per_query": candidate_rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generative PULSE evaluation")
    p.add_argument("--task", choices=sorted(TASK_SPECS), default="agnews_headline")
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disc-queries", type=int, default=2)
    p.add_argument("--eval-queries", type=int, default=3)
    p.add_argument("--candidate-pool-size", type=int, default=6)
    p.add_argument(
        "--disc-candidate-pool-size",
        type=int,
        default=None,
        help="Candidate sets per discovery query; defaults to --candidate-pool-size.",
    )
    p.add_argument("--pool-size", type=int, default=48)
    p.add_argument(
        "--pool-start",
        type=int,
        default=-1,
        help="Override pool start index in train_pairs. Default (-1) computes "
        "pool_start = disc_queries + disc_queries * disc_cands, which slides "
        "with --disc-queries; pass a positive value to keep pool fixed across "
        "different --disc-queries settings.",
    )
    p.add_argument("--n-shot", type=int, default=2)
    p.add_argument("--target-words", type=int, default=8)
    p.add_argument("--max-body-words", type=int, default=80)
    p.add_argument("--topk-each-sign", type=int, default=64)
    p.add_argument("--min-utility-delta", type=float, default=0.02)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument(
        "--signed-w",
        action="store_true",
        help="Use signed PULSE weights instead of magnitude-only weighted cosine.",
    )
    p.add_argument(
        "--retrieval-score",
        choices=["blend", "fpw", "softd2_fpw"],
        default="softd2_fpw",
        help="PULSE retrieval score: blend uses blend_03+greedy; fpw uses FPW-compatible scoring; softd2_fpw uses FPW scores with softD2 quotas.",
    )
    p.add_argument(
        "--retrieval-shortlist",
        type=int,
        default=64,
        help="Top-S pulse scores considered by greedy retrieval.",
    )
    p.add_argument("--softd2-mode-topk", type=int, default=8)
    p.add_argument("--softd2-tilt", type=int, default=1)
    p.add_argument(
        "--lambda-r",
        type=float,
        default=0.3,
        help="SAE cosine redundancy penalty for pulse greedy retrieval.",
    )
    p.add_argument(
        "--prompt-order",
        choices=["as_selected", "score_desc", "score_asc", "semantic_flow"],
        default="as_selected",
        help="Label-free ordering for pulse retrieval exemplars before utility/generation.",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--gen-batch-size",
        type=int,
        default=0,
        help="Chunk size for downstream batched generation; 0 = no chunking (legacy behavior).",
    )
    p.add_argument(
        "--utility-mode",
        choices=["logprob", "sampled_em"],
        default="logprob",
        help="Discovery utility: teacher-forced reference logprob delta, or sampled-answer EM delta (gsm8k recommended).",
    )
    p.add_argument(
        "--disc-max-new-tokens",
        type=int,
        default=0,
        help="Max new tokens for sampled_em utility during discovery; defaults to --max-new-tokens.",
    )
    p.add_argument("--skip-downstream-generation", action="store_true")
    p.add_argument("--output", default="experiments/generative/generative_eval.json")
    args = p.parse_args()

    t0 = time.time()
    print(f"Loading model {args.model}...", flush=True)
    model = load_model(args.model, args.hf_cache)
    print(f"Loading SAE {args.sae_release} layer {args.layer}...", flush=True)
    sae, _hook_name, sae_acts_name = load_sae_for_layer(args.sae_release, args.layer, hf_cache_dir=args.hf_cache)
    token_len = TokenLengthCache(model)

    rng = pyrandom.Random(args.seed)
    train_pairs, test_pairs, spec = load_task_pairs(
        args.task,
        target_words=args.target_words,
        max_body_words=args.max_body_words,
        data_root=args.data_root,
    )
    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)

    disc_candidate_pool_size = args.disc_candidate_pool_size or args.candidate_pool_size
    need_train = args.disc_queries + args.disc_queries * disc_candidate_pool_size + args.pool_size
    if len(train_pairs) < need_train:
        raise RuntimeError(f"Not enough generation pairs: have {len(train_pairs)}, need {need_train}")

    disc_queries = train_pairs[: args.disc_queries]
    disc_exemplars = train_pairs[args.disc_queries : args.disc_queries + args.disc_queries * disc_candidate_pool_size]
    auto_pool_start = args.disc_queries + args.disc_queries * disc_candidate_pool_size
    pool_start = args.pool_start if args.pool_start >= 0 else auto_pool_start
    if pool_start + args.pool_size > len(train_pairs):
        raise RuntimeError(
            f"pool_start={pool_start} + pool_size={args.pool_size} exceeds train_pairs len {len(train_pairs)}"
        )
    pool = train_pairs[pool_start : pool_start + args.pool_size]
    print(f"[Pool] pool_start={pool_start} (auto={auto_pool_start})", flush=True)
    eval_queries = test_pairs[: args.eval_queries]
    print(
        f"Splits: disc_q={len(disc_queries)} disc_ex={len(disc_exemplars)} "
        f"pool={len(pool)} eval={len(eval_queries)} k={args.n_shot} "
        f"disc_cands={disc_candidate_pool_size} eval_cands={args.candidate_pool_size}",
        flush=True,
    )

    disc_max_new = args.disc_max_new_tokens or args.max_new_tokens
    print(f"[Discovery] generative utility PULSE (mode={args.utility_mode}, disc_gen_tokens={disc_max_new})...", flush=True)
    weight, diag = discover_generative_pulse(
        model=model,
        sae=sae,
        sae_acts_name=sae_acts_name,
        token_len=token_len,
        disc_queries=disc_queries,
        disc_exemplars=disc_exemplars,
        spec=spec,
        n_shot=args.n_shot,
        candidate_pool_size=disc_candidate_pool_size,
        topk_each_sign=args.topk_each_sign,
        min_utility_delta=args.min_utility_delta,
        seed=args.seed,
        utility_mode=args.utility_mode,
        gen_max_new_tokens=disc_max_new,
    )
    print(f"[Discovery] nnz={diag['nnz']} pairs={diag['pair_count']}", flush=True)

    print("[Pool] computing exemplar vectors...", flush=True)
    pool_vectors = compute_pool_vectors(
        model=model,
        sae=sae,
        sae_acts_name=sae_acts_name,
        token_len=token_len,
        pool=pool,
        spec=spec,
    )
    print("[Eval] candidate-set and retrieval smoke...", flush=True)
    metrics = evaluate(
        model=model,
        sae=sae,
        sae_acts_name=sae_acts_name,
        token_len=token_len,
        pulse_weight=weight,
        pool=pool,
        pool_vectors=pool_vectors,
        eval_queries=eval_queries,
        spec=spec,
        n_shot=args.n_shot,
        n_candidates=args.candidate_pool_size,
        beta=args.beta,
        signed_w=args.signed_w,
        retrieval_score=args.retrieval_score,
        retrieval_shortlist=args.retrieval_shortlist,
        softd2_mode_topk=args.softd2_mode_topk,
        softd2_tilt=args.softd2_tilt,
        lambda_r=args.lambda_r,
        prompt_order=args.prompt_order,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        run_downstream=not args.skip_downstream_generation,
        gen_batch_size=args.gen_batch_size,
    )

    payload = {
        "task": spec["name"],
        "task_arg": args.task,
        "utility": (
            "sampled_answer_em_delta"
            if args.utility_mode == "sampled_em"
            else "teacher_forced_reference_logprob_delta"
        ),
        "utility_mode": args.utility_mode,
        "disc_max_new_tokens": disc_max_new,
        "model": args.model,
        "sae_release": args.sae_release,
        "layer": args.layer,
        "seed": args.seed,
        "scale": {
            "disc_queries": args.disc_queries,
            "eval_queries": args.eval_queries,
            "candidate_pool_size": args.candidate_pool_size,
            "disc_candidate_pool_size": disc_candidate_pool_size,
            "pool_size": args.pool_size,
            "n_shot": args.n_shot,
            "target_words": args.target_words,
            "topk_each_sign": args.topk_each_sign,
            "beta": args.beta,
            "signed_w": args.signed_w,
            "retrieval_score": args.retrieval_score,
            "selection_policy": (
                "fpw_signed_fusion"
                if args.retrieval_score == "fpw" and args.signed_w
                else args.retrieval_score
            ),
            "retrieval_shortlist": args.retrieval_shortlist,
            "softd2_mode_topk": args.softd2_mode_topk,
            "softd2_tilt": args.softd2_tilt,
            "lambda_r": args.lambda_r,
            "prompt_order": args.prompt_order,
            "max_new_tokens": args.max_new_tokens,
            "downstream_generation": not args.skip_downstream_generation,
        },
        "discovery": diag,
        "metrics": metrics,
        "elapsed_min": round((time.time() - t0) / 60.0, 2),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metric_summary = {
        key: value
        for key, value in payload["metrics"].items()
        if key not in {"generations", "per_query"}
    }
    print(json.dumps(metric_summary, indent=2), flush=True)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
