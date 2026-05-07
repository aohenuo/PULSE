#!/usr/bin/env python3
"""Layer B retrieval accuracy evaluation.

This is the clean release entry point for the retrieval experiment:

  - pulse: canonical blend_03 PULSE retrieval
  - pulse with --retrieval-score softd2_fpw: PULSE FPW scores plus softD2 quotas

The script supports either inline PULSE discovery or a precomputed payload.
"""
from __future__ import annotations

import argparse
import json
import random as pyrandom
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pulse.data import prepare_dataset, select_eval_examples
from pulse.intervention import clear_gpu, rebuild_weight as _rebuild_weight
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
from pulse.pulse_scaling import _compute_static_vectors, _sae_acts_mean
from pulse.scoring import fpw_scores, score_blend03, select_retrieval_greedy, select_softd2_fpw
from pulse.w_cosine import WCosineConfig, discover_pulse_weight


DEFAULT_METHODS = ["pulse"]
SUPPORTED_METHODS = set(DEFAULT_METHODS)


def _validate_methods(methods: list[str]) -> None:
    unknown = sorted(set(methods) - SUPPORTED_METHODS)
    if unknown:
        raise ValueError(
            f"Unsupported Layer B methods: {unknown}. "
            f"Supported: {sorted(SUPPORTED_METHODS)}"
        )


def _make_exs_from_indices(pool, indices, label_words):
    return [
        (
            str(pool[i][0]),
            str(pool[i][1]),
            label_words.get(str(pool[i][1]), str(pool[i][1])),
        )
        for i in indices
    ]


def _label_distribution(rows) -> dict[str, int]:
    return dict(sorted(Counter(str(label) for _text, label in rows).items()))


def _eval_ice_set(model, exs, query_text, gold_label, label_words, instruction, template_id):
    prompt, _ = build_prompt_with_segments(instruction, exs, query_text, template_id)
    scores = score_candidate_label_sequences(model, prompt, label_words)
    pred, _ = predict_label_and_confidence_gap_from_scores(scores)
    margin = margin_for_gold_label_scores(scores, gold_label)
    return int(pred == gold_label), float(margin), str(pred)


def learn_pulse_inline(
    *,
    model,
    sae,
    sae_acts_name: str,
    token_len: TokenLengthCache,
    train_data: list[tuple[str, str]],
    label_words: dict[str, str],
    instruction: str,
    n_shot: int,
    seed: int,
    disc_queries: int,
    candidate_pool_size: int,
    template_id: str,
    topk_each_sign: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Learn a sparse PULSE weight from discovery queries and donor exemplars."""
    rng = pyrandom.Random(seed)
    shuffled = list(train_data)
    rng.shuffle(shuffled)
    disc_q_subset = shuffled[:disc_queries]
    disc_ex_subset = shuffled[disc_queries:]

    cfg = WCosineConfig(
        disc_queries=disc_queries,
        candidate_pool_size=candidate_pool_size,
        disc_n_shot=n_shot,
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

    weight, _f_plus, _f_minus, diag = discover_pulse_weight(
        model=model,
        sae=sae,
        sae_acts_name=sae_acts_name,
        token_len=token_len,
        disc_queries=disc_q_subset,
        disc_exemplars=disc_ex_subset,
        label_words=label_words,
        instruction=instruction,
        cfg=cfg,
    )
    return weight, diag


def evaluate_accuracy(
    dataset: str,
    seed: int = 42,
    n_test_queries: int = 512,
    candidate_pool_size: int = 48,
    n_shot_total: int = 4,
    template_id: str = "default",
    model_name: str = "google/gemma-2-2b",
    sae_release: str = "gemma-scope-2b-pt-res-canonical",
    hf_cache_dir: str | None = None,
    utility_payload: dict | None = None,
    data_root: str | None = None,
    selection_methods: list[str] | None = None,
    layer: int | None = None,
    disc_queries: int = 64,
    topk_each_sign: int = 256,
    max_pool_size: int = 2000,
    beta: float = 0.3,
    shortlist: int = 200,
    retrieval_score: str = "softd2_fpw",
    softd2_mode_topk: int = 8,
    softd2_tilt: int = 1,
    preloaded_model=None,
) -> dict[str, Any]:
    methods = list(selection_methods or DEFAULT_METHODS)
    _validate_methods(methods)
    if retrieval_score not in {"blend", "softd2_fpw"}:
        raise ValueError(f"Unsupported retrieval_score: {retrieval_score}")

    if utility_payload is not None:
        layer = layer if layer is not None else int(utility_payload["selected_layer"])
        pulse_weight = _rebuild_weight(utility_payload, int(layer))
        discovery_diag = {"source": "payload"}
    else:
        layer = int(layer if layer is not None else DEFAULT_LAYER.get(dataset, 12))
        pulse_weight = None
        discovery_diag = {"source": "inline"}

    if preloaded_model is not None:
        model = preloaded_model
    else:
        print(f"  Loading model {model_name}...", flush=True)
        model = load_model(model_name, hf_cache_dir)
    sae, _hook_name, sae_acts_name = load_sae_for_layer(
        sae_release,
        int(layer),
        hf_cache_dir=hf_cache_dir,
    )
    token_len = TokenLengthCache(model)

    print(
        f"  Layer={layer}, seed={seed}, eval={n_test_queries}, "
        f"pool={max_pool_size}, discovery_candidates={candidate_pool_size}, k={n_shot_total}",
        flush=True,
    )

    train_data, test_data, label_words, instruction = prepare_dataset(
        dataset,
        seed,
        data_root=data_root,
    )
    eval_queries = [
        q for q in select_eval_examples(test_data, seed=seed, max_queries=n_test_queries)
        if str(q[1]) in label_words
    ]

    rng_pool = pyrandom.Random(seed + 12345)
    if max_pool_size > 0 and len(train_data) > max_pool_size:
        pool = rng_pool.sample(train_data, max_pool_size)
    else:
        pool = list(train_data)
    pool_labels = [str(row[1]) for row in pool]
    print(f"  Pool size: {len(pool)}, eval queries: {len(eval_queries)}", flush=True)

    if pulse_weight is None and "pulse" in methods:
        print(f"  [Discovery] Learning PULSE weight inline ({disc_queries} queries)...", flush=True)
        t_disc = time.time()
        pulse_weight, discovery_diag = learn_pulse_inline(
            model=model,
            sae=sae,
            sae_acts_name=sae_acts_name,
            token_len=token_len,
            train_data=train_data,
            label_words=label_words,
            instruction=instruction,
            n_shot=n_shot_total,
            seed=seed,
            disc_queries=disc_queries,
            candidate_pool_size=candidate_pool_size,
            template_id=template_id,
            topk_each_sign=topk_each_sign,
        )
        discovery_diag["elapsed_sec"] = round(time.time() - t_disc, 1)

    print("  [Setup] Computing SAE vectors for pool...", flush=True)
    static_sae = _compute_static_vectors(model, sae, sae_acts_name, pool, template_id)

    results = {m: {"correct": [], "margins": []} for m in methods}
    ice_label_counts = {m: Counter() for m in methods}
    print(f"\n  [Eval] {len(eval_queries)} queries...", flush=True)
    t0 = time.time()

    for q_idx, (query_text, gold_label_raw) in enumerate(eval_queries):
        gold_label = str(gold_label_raw)
        if gold_label not in label_words:
            continue

        zero_prompt, _ = build_prompt_with_segments(instruction, [], query_text, template_id)
        q_sae = _sae_acts_mean(model, sae, sae_acts_name, zero_prompt)
        sel_results: dict[str, tuple[float, float, list[int]]] = {}

        if "pulse" in methods and pulse_weight is not None:
            active = pulse_weight.nonzero(as_tuple=True)[0]
            if active.numel() > 0:
                if retrieval_score == "softd2_fpw":
                    scores = fpw_scores(
                        q_sae.cpu(),
                        static_sae.cpu(),
                        active.cpu(),
                        pulse_weight.cpu(),
                    )
                    idx_list = select_softd2_fpw(
                        scores=scores,
                        pool_labels=pool_labels,
                        n_shot=n_shot_total,
                        mode_topk=softd2_mode_topk,
                        tilt=softd2_tilt,
                    )
                else:
                    blend_scores = score_blend03(
                        q_sae.cpu(),
                        static_sae.cpu(),
                        active.cpu(),
                        pulse_weight.cpu(),
                        beta=beta,
                    )
                    idx_list = select_retrieval_greedy(
                        blend_scores=blend_scores,
                        pool_sae=static_sae.cpu(),
                        pool_labels=pool_labels,
                        active_features=active.cpu(),
                        n_shot=n_shot_total,
                        shortlist=min(shortlist, len(pool)),
                        lambda_r=0.3,
                        lambda_b=0.0,
                    )
                exs = _make_exs_from_indices(pool, idx_list, label_words)
                ok, margin, _ = _eval_ice_set(
                    model,
                    exs,
                    query_text,
                    gold_label,
                    label_words,
                    instruction,
                    template_id,
                )
                sel_results["pulse"] = (float(ok), float(margin), idx_list)

        for method, (ok, margin, idx_list) in sel_results.items():
            results[method]["correct"].append(float(ok))
            results[method]["margins"].append(float(margin))
            for idx in idx_list:
                ice_label_counts[method][pool_labels[idx]] += 1

        if (q_idx + 1) % 20 == 0:
            n = q_idx + 1
            line = f"    [{n:3d}/{len(eval_queries)}] ({(time.time() - t0) / 60:.0f}m)"
            for method in methods:
                correct = results[method]["correct"]
                if correct:
                    line += f"  {method}={sum(correct) / len(correct):.3f}"
            print(line, flush=True)

    elapsed = time.time() - t0
    accuracy = {}
    for method in methods:
        correct = results[method]["correct"]
        margins = results[method]["margins"]
        if correct:
            accuracy[method] = {
                "accuracy": round(sum(correct) / len(correct), 4),
                "mean_margin": round(sum(margins) / len(margins), 4),
                "n": len(correct),
                "ice_label_distribution": dict(sorted(ice_label_counts[method].items())),
            }

    return {
        "dataset": dataset,
        "model": model_name,
        "sae_release": sae_release,
        "layer": int(layer),
        "seed": seed,
        "n_shot": n_shot_total,
        "discovery_candidates": candidate_pool_size,
        "pool_size": len(pool),
        "n_eval": max((len(results[m]["correct"]) for m in methods), default=0),
        "elapsed_min": round(elapsed / 60, 1),
        "accuracy": accuracy,
        "methods": methods,
        "retrieval_score": retrieval_score,
        "softd2_fpw": {
            "mode_topk": int(softd2_mode_topk),
            "tilt": int(softd2_tilt),
        },
        "discovery": discovery_diag,
        "pool_label_distribution": _label_distribution(pool),
        "train_label_distribution": _label_distribution(train_data),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer B retrieval accuracy evaluation")
    parser.add_argument("--datasets", nargs="+", default=["agnews", "rest14", "lap14", "emoc"])
    parser.add_argument("--model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-2b-pt-res-canonical")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--hf_cache", type=str, default=None)
    parser.add_argument("--utility-payload", type=str, default=None)
    parser.add_argument("--disc-queries", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-queries", type=int, default=512)
    parser.add_argument("--pool-size", type=int, default=2000)
    parser.add_argument("--n-cand", type=int, default=48, help="Candidate sets per discovery query.")
    parser.add_argument("--topk-each-sign", type=int, default=256)
    parser.add_argument("--n-shot", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--shortlist", type=int, default=200)
    parser.add_argument(
        "--retrieval-score",
        choices=["blend", "softd2_fpw"],
        default="softd2_fpw",
        help="PULSE retrieval policy: blend uses blend_03+greedy; softd2_fpw uses FPW scores with softD2 label quotas.",
    )
    parser.add_argument("--softd2-mode-topk", type=int, default=8)
    parser.add_argument("--softd2-tilt", type=int, default=1)
    parser.add_argument("--methods", nargs="+", default=None, choices=sorted(SUPPORTED_METHODS))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="experiments/layer_b")
    args = parser.parse_args()

    methods = list(args.methods or DEFAULT_METHODS)
    try:
        _validate_methods(methods)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    payload = None
    if args.utility_payload:
        with open(args.utility_payload) as handle:
            payload = json.load(handle)

    print(f"Loading {args.model} once for all datasets...", flush=True)
    model = load_model(args.model, args.hf_cache)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for dataset in args.datasets:
        print(f"\n{'=' * 70}\n=== {dataset}  k={args.n_shot}  layer={args.layer}\n{'=' * 70}")
        result = evaluate_accuracy(
            dataset=dataset,
            seed=args.seed,
            n_test_queries=args.eval_queries,
            candidate_pool_size=args.n_cand,
            n_shot_total=args.n_shot,
            model_name=args.model,
            sae_release=args.sae_release,
            hf_cache_dir=args.hf_cache,
            utility_payload=payload,
            data_root=args.data_root,
            selection_methods=methods,
            layer=args.layer,
            disc_queries=args.disc_queries,
            max_pool_size=args.pool_size,
            topk_each_sign=args.topk_each_sign,
            beta=args.beta,
            shortlist=args.shortlist,
            retrieval_score=args.retrieval_score,
            softd2_mode_topk=args.softd2_mode_topk,
            softd2_tilt=args.softd2_tilt,
            preloaded_model=model,
        )
        all_results[dataset] = result

        layer_tag = f"_L{args.layer}" if args.layer is not None else ""
        score_tag = f"_{args.retrieval_score}" if args.retrieval_score != "blend" else ""
        per_ds_path = out_dir / f"layer_b_{dataset}_s{args.seed}{layer_tag}_k{args.n_shot}{score_tag}.json"
        with open(per_ds_path, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nSaved: {per_ds_path}")

        print(f"\n=== {dataset} k={args.n_shot} Summary ===")
        for method, item in sorted(
            result["accuracy"].items(),
            key=lambda kv: kv[1]["accuracy"],
            reverse=True,
        ):
            print(f"  {method:<10} acc={item['accuracy']:.4f}  margin={item['mean_margin']:+.4f}")
        clear_gpu()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(all_results, handle, indent=2, default=str)
        print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
