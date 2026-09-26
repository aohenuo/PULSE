"""Portable JSON interface for precomputed SAE representations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from . import __version__
from .configs import RetrievalConfig
from .core import (
    learn_pulse_weight, select_pulse_context,
    select_classification_context, select_generation_context,
)


def select_from_payload(payload: dict, config: RetrievalConfig, method: str) -> dict:
    """Learn or reuse a weight, then select ordered pool row indices."""
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    if method not in {"pulse", "classification-fusion", "generation-fusion"}:
        raise ValueError("unknown selection method")
    if "pulse_weight" in payload and "discovery" in payload:
        raise ValueError("provide pulse_weight or discovery, not both")

    def tensor(value):
        result = torch.tensor(value, dtype=torch.float32)
        if not torch.isfinite(result).all():
            raise ValueError("input tensors must contain finite numbers")
        return result

    if "pulse_weight" in payload:
        weight = tensor(payload["pulse_weight"])
    else:
        discovery = payload["discovery"]
        if (not isinstance(discovery, list) or not discovery
                or any(not isinstance(row, dict) for row in discovery)):
            raise ValueError("discovery must be a nonempty array of objects")
        topk = payload["topk_each_sign"]
        if isinstance(topk, bool) or not isinstance(topk, int) or topk < 0:
            raise ValueError("topk_each_sign must be a nonnegative integer")
        weight = learn_pulse_weight(
            [tensor(row["utilities"]) for row in discovery],
            [tensor(row["activations"]) for row in discovery],
            topk_each_sign=topk,
        )
    query, pool = tensor(payload["query_sae"]), tensor(payload["pool_sae"])
    if method == "pulse":
        selected = select_pulse_context(query, pool, weight, config=config)
    else:
        selector = (select_classification_context if method == "classification-fusion"
                    else select_generation_context)
        selected = selector(query, tensor(payload["query_sbert"]), pool,
                            tensor(payload["pool_sbert"]), weight, config=config)
    result = {
        "schema_version": 1, "package_version": __version__, "method": method,
        "config": asdict(config), "indices": selected,
        "pulse_weight": weight.tolist(),
        "active_features": int(torch.count_nonzero(weight)),
    }
    if "pool_ids" in payload:
        ids = payload["pool_ids"]
        if len(ids) != pool.shape[0] or len(set(ids)) != len(ids):
            raise ValueError("pool_ids must be unique and match the pool row count")
        result["selected_ids"] = [ids[i] for i in selected]
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON tensor input (see examples/tiny.json)")
    parser.add_argument("--output", type=Path, help="Write JSON; default: stdout")
    parser.add_argument("--method", choices=["pulse", "classification-fusion", "generation-fusion"], default="pulse")
    parser.add_argument("--n-shot", type=int, default=4)
    parser.add_argument("--shortlist", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda-r", type=float, default=0.3)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        payload = json.loads(raw)
        config = RetrievalConfig(alpha=args.alpha, beta=args.beta, n_shot=args.n_shot,
                                 shortlist=args.shortlist, lambda_r=args.lambda_r)
        result = select_from_payload(payload, config, args.method)
        result["input_sha256"] = hashlib.sha256(raw).hexdigest()
        output = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
