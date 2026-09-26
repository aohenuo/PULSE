"""Run the five baselines with one LM and shared, target-hidden feature caches.

All remaining arguments are pulse-eval arguments. Outputs live at OUTPUT/METHOD.
Caches are in memory for this invocation; they never load unverifiable old tensors.
"""
from __future__ import annotations

import argparse
from copy import copy
import json
from pathlib import Path

import torch

from pceu_rtr.baselines import BASELINE_METHODS
from pceu_rtr.evaluate import ModelBackend, parse_args, run


class CachedBackend(ModelBackend):
    def __init__(self, args):
        super().__init__(args)
        self._features = {}
        self._semantics = {}

    def encode(self, prompt):
        if prompt not in self._features:
            self._features[prompt] = super().encode(prompt)
        return self._features[prompt]

    def semantic_encode(self, texts):
        missing = list(dict.fromkeys(text for text in texts if text not in self._semantics))
        if missing:
            values = super().semantic_encode(missing)
            self._semantics.update(zip(missing, values))
        return torch.stack([self._semantics[text] for text in texts])


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--methods", nargs="+", choices=BASELINE_METHODS, default=list(BASELINE_METHODS))
    suite, rest = parser.parse_known_args()
    args = parse_args(rest + ["--method", suite.methods[0]])
    root = args.output
    if root.exists() and any(root.iterdir()):
        raise SystemExit("suite output directory must be empty")
    backend_args = copy(args)
    if any(method in {"knn_sae", "lex_sim"} for method in suite.methods):
        backend_args.method = "knn_sae"
    torch.set_num_threads(4)
    backend = CachedBackend(backend_args)
    summaries = []
    for method in suite.methods:
        current = copy(args)
        current.method = method
        current.output = root / method
        print(f"Starting {method}", flush=True)
        summaries.append(run(current, backend))
    (root / "suite.json").write_text(json.dumps({
        "status": "complete", "methods": suite.methods,
        "task": args.task, "summaries": summaries,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
