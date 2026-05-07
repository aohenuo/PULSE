#!/usr/bin/env python3
"""Layer A candidate-set scoring with canonical blend_03.

Formal mainline for the paper's first downstream use:
  1. Learn PULSE weight W on discovery queries
  2. Score sampled candidate demonstration sets with `score_set(...)`
  3. Pick the top-ranked set under PULSE
  4. Measure actual ICL accuracy / margin on held-out queries

Unlike the removed legacy pipeline, this script does not use `W·Δa` scoring.
The PULSE method is the set version of the canonical blend_03 score.
"""
from __future__ import annotations

import argparse
import json
import random as pyrandom
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

# Avoid CPU oversubscription when multiple jobs run on the same node.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

_PROJ = Path(__file__).resolve().parents[2]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from pulse.data import prepare_dataset
from pulse.metrics import (
    margin_for_gold_label_scores,
    predict_label_and_confidence_gap_from_scores,
    score_candidate_label_sequences,
)
from pulse.modeling import (
    DEFAULT_LAYER,
    TokenLengthCache,
    build_prompt_with_segments,
    load_model,
    load_sae_for_layer,
)
from pulse.pulse_scaling import (
    ScalingConfig,
    _compute_static_vectors,
    _sae_acts_mean,
    _split_data,
    _zero_shot_prompt,
)
from pulse.scoring import score_blend03
from pulse.stats import bootstrap_ci
from pulse.w_cosine import WCosineConfig, discover_pulse_weight


DEFAULT_METHODS: list[str] = ["pulse"]
SUPPORTED_METHODS = set(DEFAULT_METHODS)


def macro_f1(true_labels: Sequence[str], pred_labels: Sequence[str]) -> float:
    labels = sorted(set(true_labels) | set(pred_labels))
    if not labels:
        return 0.0
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))


def _make_exs_from_indices(pool, indices, label_words):
    return [
        (str(pool[i][0]), str(pool[i][1]), label_words.get(str(pool[i][1]), str(pool[i][1])))
        for i in indices
    ]


def _eval_ice_set(model, exs, query_text, gold_label, label_words, instruction, template_id):
    prompt, _ = build_prompt_with_segments(instruction, exs, query_text, template_id)
    scores = score_candidate_label_sequences(model, prompt, label_words)
    pred, _ = predict_label_and_confidence_gap_from_scores(scores)
    margin = margin_for_gold_label_scores(scores, gold_label)
    return int(pred == gold_label), float(margin), str(pred)


def evaluate_selection_accuracy(
    *,
    dataset: str,
    seed: int = 42,
    n_test_queries: int = 512,
    n_candidates: int = 48,
    n_shot_total: int = 4,
    template_id: str = "default",
    model_name: str = "google/gemma-2-2b",
    sae_release: str = "gemma-scope-2b-pt-res-canonical",
    hf_cache_dir: str | None = None,
    data_root: str | None = None,
    layer: int | None = None,
    disc_queries: int = 64,
    disc_candidate_pool_size: int = 32,
    topk_each_sign: int = 256,
    max_pool_size: int = 2000,
    beta: float = 0.3,
    selection_methods: list[str] | None = None,
    preloaded_model=None,
) -> dict[str, Any]:
    methods = list(selection_methods or DEFAULT_METHODS)
    unknown = sorted(set(methods) - SUPPORTED_METHODS)
    if unknown:
        raise ValueError(
            f"Unsupported Layer A methods in clean PULSE release: {unknown}. "
            f"Supported: {sorted(SUPPORTED_METHODS)}"
        )
    layer = int(layer if layer is not None else DEFAULT_LAYER.get(dataset, 12))

    if preloaded_model is not None:
        model = preloaded_model
    else:
        print(f"  Loading model {model_name}...", flush=True)
        model = load_model(model_name, hf_cache_dir)
    token_len = TokenLengthCache(model)
    sae, _hook_name, sae_acts_name = load_sae_for_layer(
        sae_release, layer, hf_cache_dir=hf_cache_dir
    )

    print(
        f"  Layer={layer}, seed={seed}, eval={n_test_queries}, pool={max_pool_size}, "
        f"n_candidates={n_candidates}, k={n_shot_total}",
        flush=True,
    )

    train_data, test_data, label_words, instruction = prepare_dataset(
        dataset, seed, data_root=data_root,
    )
    cfg = ScalingConfig(
        mode="S",
        seed=seed,
        n_shot=n_shot_total,
        eval_queries=n_test_queries,
        disc_queries=disc_queries,
        candidate_pool_size=disc_candidate_pool_size,
        max_pool_size=max_pool_size,
        hf_cache_dir=hf_cache_dir,
    )
    cfg.layer = layer
    if data_root:
        cfg.data_root = data_root

    _calib_unused, disc_query_rows, disc_exemplars, pool, eval_queries = _split_data(
        train_data, test_data, cfg
    )
    eval_queries = [q for q in eval_queries if str(q[1]) in label_words]
    print(
        f"  Splits: disc_q={len(disc_query_rows)}  disc_ex={len(disc_exemplars)}  "
        f"pool={len(pool)}  eval={len(eval_queries)}",
        flush=True,
    )

    # ── Stage I: discover sparse W ──
    print(f"  [Discovery] canonical blend_03 set-scoring weight...", flush=True)
    wc_cfg = WCosineConfig(
        disc_queries=disc_queries,
        candidate_pool_size=disc_candidate_pool_size,
        disc_n_shot=n_shot_total,
        topk_each_sign=topk_each_sign,
        max_topk_each_sign=max(topk_each_sign * 2, 512),
        min_utility_delta=0.1,
        use_magnitude_weighting=True,
        use_residualized=True,
        use_variance_norm=True,
        min_matched_pair_fraction=0.5,
        min_matched_pairs=64,
        max_pairs_per_query=256,
        min_effective_pairs_per_query=64,
        pooling="mean",
        seed=seed,
        template_id=template_id,
    )
    t_disc = time.time()
    pulse_weight, _f_plus, _f_minus, diag = discover_pulse_weight(
        model=model,
        sae=sae,
        sae_acts_name=sae_acts_name,
        token_len=token_len,
        disc_queries=disc_query_rows,
        disc_exemplars=disc_exemplars,
        label_words=label_words,
        instruction=instruction,
        cfg=wc_cfg,
    )
    del _f_plus, _f_minus
    if diag is None:
        raise RuntimeError("Discovery diagnostics unexpectedly missing")
    active = pulse_weight.nonzero(as_tuple=True)[0]
    print(
        f"  [Discovery] nnz={int(active.numel())}  pairs={diag['pair_count']}  "
        f"time={time.time() - t_disc:.0f}s",
        flush=True,
    )

    # ── Pool vectors for set scoring ──
    print(f"  [Pool] SAE vectors for {len(pool)} exemplars...", flush=True)
    static_sae = _compute_static_vectors(model, sae, sae_acts_name, pool, template_id)

    results = {
        m: {"correct": [], "margins": [], "true": [], "pred": []}
        for m in methods
    }

    print(f"  [Eval] {len(eval_queries)} queries × {n_candidates} candidate sets...", flush=True)
    t0 = time.time()
    for qi, (query_text, gold_label_raw) in enumerate(eval_queries):
        gold_label = str(gold_label_raw)
        if gold_label not in label_words:
            continue

        z_prompt = _zero_shot_prompt(instruction, query_text, template_id)
        q_sae = _sae_acts_mean(model, sae, sae_acts_name, z_prompt)

        rng = pyrandom.Random(seed + qi * 10000)
        candidates = [rng.sample(range(len(pool)), n_shot_total) for _ in range(n_candidates)]

        # Pool-wide blend_03 z-normed once per query (paper semantics).
        # Per-set PULSE score = mean of pool_blend at set's indices.
        if active.numel() > 0:
            pool_blend = score_blend03(
                q_sae.cpu(), static_sae.cpu(), active.cpu(), pulse_weight.cpu(), beta=beta
            )
        else:
            pool_blend = None

        pulse_scores: list[float] = []

        for idxs in candidates:
            if pool_blend is not None:
                pulse_scores.append(float(pool_blend[torch.as_tensor(idxs)].mean().item()))
            else:
                pulse_scores.append(0.0)

        eval_cache: dict[int, tuple[int, float, str]] = {}

        def _eval_candidate(ci: int) -> tuple[int, float, str]:
            if ci not in eval_cache:
                eval_cache[ci] = _eval_ice_set(
                    model,
                    _make_exs_from_indices(pool, candidates[ci], label_words),
                    query_text,
                    gold_label,
                    label_words,
                    instruction,
                    template_id,
                )
            return eval_cache[ci]

        sel = {"pulse": max(range(n_candidates), key=lambda i: pulse_scores[i])}

        for method, ci in sel.items():
            ok, margin, pred = _eval_candidate(ci)
            results[method]["correct"].append(float(ok))
            results[method]["margins"].append(float(margin))
            results[method]["true"].append(gold_label)
            results[method]["pred"].append(pred)

        if (qi + 1) % 20 == 0:
            n_done = qi + 1
            line = f"    [{n_done:3d}/{len(eval_queries)}] ({(time.time()-t0)/60:.0f}m)"
            for m in methods:
                corr = results[m]["correct"]
                if corr:
                    line += f"  {m[:8]}={sum(corr)/len(corr):.3f}"
            print(line, flush=True)

    elapsed = time.time() - t0
    accuracy = {}
    for m in methods:
        corr = results[m]["correct"]
        mg = results[m]["margins"]
        if not corr:
            continue
        mean_acc, ci_lo, ci_hi = bootstrap_ci(corr, n_bootstrap=2000, seed=seed)
        accuracy[m] = {
            "accuracy": round(sum(corr) / len(corr), 4),
            "macro_f1": round(macro_f1(results[m]["true"], results[m]["pred"]), 4),
            "mean_margin": round(sum(mg) / len(mg), 4),
            "acc_ci_low": round(ci_lo, 4),
            "acc_ci_high": round(ci_hi, 4),
            "n": len(corr),
        }

    return {
        "dataset": dataset,
        "paradigm": "candidate_set_scoring",
        "model": model_name,
        "sae_release": sae_release,
        "layer": layer,
        "seed": seed,
        "n_shot": n_shot_total,
        "n_candidates": n_candidates,
        "disc_queries": len(disc_query_rows),
        "disc_candidate_pool_size": disc_candidate_pool_size,
        "pool_size": len(pool),
        "n_eval": len(results[methods[0]]["correct"]) if methods else 0,
        "elapsed_min": round(elapsed / 60, 1),
        "discovery": {
            "pair_count": int(diag["pair_count"]),
            "nnz": int(active.numel()),
        },
        "accuracy": accuracy,
        "methods": methods,
    }


def main():
    p = argparse.ArgumentParser(description="Layer A candidate-set scoring with blend_03")
    p.add_argument("--datasets", nargs="+", default=["agnews", "rest14", "lap14", "emoc"])
    p.add_argument("--model", type=str, default="google/gemma-2-2b")
    p.add_argument("--sae_release", type=str, default="gemma-scope-2b-pt-res-canonical")
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--template-id", type=str, default="default")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--hf_cache", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disc-queries", type=int, default=64)
    p.add_argument("--disc-candidate-pool", type=int, default=32)
    p.add_argument("--eval-queries", type=int, default=512)
    p.add_argument("--pool-size", type=int, default=2000)
    p.add_argument("--n-cand", type=int, default=48,
                   help="Number of sampled k-shot candidate sets per query")
    p.add_argument("--n-shot", type=int, default=4)
    p.add_argument("--topk-each-sign", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--methods", nargs="+", default=None, choices=sorted(SUPPORTED_METHODS))
    p.add_argument("--results_dir", type=str, default="experiments/T1_selection")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    methods = list(args.methods or DEFAULT_METHODS)
    unknown = sorted(set(methods) - SUPPORTED_METHODS)
    if unknown:
        raise SystemExit(
            f"Unsupported Layer A methods in clean PULSE release: {unknown}. "
            f"Supported: {sorted(SUPPORTED_METHODS)}"
        )

    print(f"Loading {args.model} once for all datasets...", flush=True)
    model = load_model(args.model, args.hf_cache)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for ds in args.datasets:
        print(f"\n{'='*70}\n=== {ds}  k={args.n_shot}  layer={args.layer}\n{'='*70}", flush=True)
        result = evaluate_selection_accuracy(
            dataset=ds,
            seed=args.seed,
            n_test_queries=args.eval_queries,
            n_candidates=args.n_cand,
            n_shot_total=args.n_shot,
            template_id=args.template_id,
            model_name=args.model,
            sae_release=args.sae_release,
            hf_cache_dir=args.hf_cache,
            data_root=args.data_root,
            layer=args.layer,
            disc_queries=args.disc_queries,
            disc_candidate_pool_size=args.disc_candidate_pool,
            topk_each_sign=args.topk_each_sign,
            max_pool_size=args.pool_size,
            beta=args.beta,
            selection_methods=methods,
            preloaded_model=model,
        )
        all_results[ds] = result

        per_ds = out_dir / f"selection_{ds}_s{args.seed}_k{args.n_shot}.json"
        with open(per_ds, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved: {per_ds}", flush=True)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nAll results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
