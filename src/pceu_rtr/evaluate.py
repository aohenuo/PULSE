"""Run PULSE on raw datasets with a causal LM and a pretrained SAE."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sysconfig
import time

import torch

from . import __version__
from .configs import RetrievalConfig
from .core import learn_pulse_weight, select_pulse_context
from .data import Example, load_examples, normalize_input


LABELS = {
    "agnews": ["World", "Sports", "Business", "Sci/Tech"],
    "rest14": ["Positive", "Negative", "Neutral"],
    "lap14": ["Positive", "Negative", "Neutral"],
    "emoc": ["angry", "happy", "others", "sad"],
}


def default_eval_data(task: str) -> Path:
    """Find the bundled evaluation TSV in a checkout or installed wheel."""
    filename = f"{task}.tsv"
    checkout = Path(__file__).resolve().parents[2] / "dataset" / filename
    if checkout.is_file():
        return checkout
    installed = Path(sysconfig.get_path("data")) / "share" / "pceu-rtr-core" / "dataset" / filename
    if installed.is_file():
        return installed
    raise FileNotFoundError(
        f"bundled evaluation data {filename} not found; pass --eval-data explicitly"
    )


def build_prompt(task: str, query: str, demos: list[Example]) -> str:
    if task in LABELS:
        instruction, source, target = "", "Input", "Label"
    elif task == "commongen":
        instruction = "Write one natural sentence that uses all of the given concepts.\n\n"
        source, target = "Concepts", "Sentence"
    elif task == "gsm8k":
        instruction = "Solve the math word problem step by step. End with a line of the form '#### <number>'.\n\n"
        source, target = "Question", "Answer"
    else:
        raise ValueError(f"unsupported task: {task}")
    if any(d.target is None for d in demos):
        raise ValueError("demonstrations require targets")
    return instruction + "".join(
        f"{source}: {d.input}\n{target}: {d.target}\n\n" for d in demos
    ) + f"{source}: {query}\n{target}:"


def numeric_answer(text: str) -> str | None:
    # Require the declared answer delimiter; avoid mistaking reasoning for an answer.
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if match is None:
        return None
    from decimal import Decimal
    return str(Decimal(match.group(1).replace(",", "")).normalize())


class _EncodingCaptured(Exception):
    pass


class ModelBackend:
    """Direct Hugging Face residual hook; never substitutes SAE reconstruction."""

    def __init__(self, args):
        if args.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sae_lens import SAE

        self.device = args.device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable in this interpreter")
        dtype = getattr(torch, args.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.model_revision, local_files_only=args.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.model_revision, torch_dtype=dtype,
            local_files_only=args.local_files_only, attn_implementation="eager",
        ).to(self.device).eval()
        result = SAE.from_pretrained(release=args.sae_release, sae_id=args.sae_id, device=self.device)
        self.sae = (result[0] if isinstance(result, tuple) else result).to(self.device).eval()
        self.layer = args.layer
        layers = getattr(self.model.model, "layers", None)
        if layers is None or not 0 <= self.layer < len(layers):
            raise ValueError("model must expose model.layers with the requested residual layer")
        hook_name = getattr(self.sae.cfg, "hook_name", None)
        if hook_name and hook_name != f"blocks.{self.layer}.hook_resid_post":
            raise ValueError(f"SAE hook {hook_name} does not match residual layer {self.layer}")
        self.max_input_tokens = args.max_input_tokens
        self.max_new_tokens = args.max_new_tokens
        self.metadata = {
            "model": args.model, "model_revision": getattr(self.model.config, "_commit_hash", args.model_revision),
            "sae_release": args.sae_release, "sae_id": args.sae_id,
            "sae_hook": hook_name, "layer": self.layer,
            "sae_width": int(self.sae.cfg.d_sae),
            "pooling": "mean_all_non_bos_prefix_tokens",
            "device": self.device, "dtype": args.dtype,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.startswith("cuda") else None,
        }
        from importlib.metadata import version
        self.metadata.update({p: version(p) for p in ["transformers", "sae-lens"]})

    def tokens(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)
        if ids.shape[1] > self.max_input_tokens:
            raise ValueError(f"prompt has {ids.shape[1]} tokens, exceeds --max-input-tokens={self.max_input_tokens}; no silent truncation")
        return ids

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        ids = self.tokens(prompt)
        captured = []

        def capture(module, inputs, output):
            residual = output[0] if isinstance(output, tuple) else output
            start = int(ids[0, 0].item() == self.tokenizer.bos_token_id)
            acts = self.sae.encode(residual[:, start:].to(next(self.sae.parameters()).dtype))
            captured.append(acts.float().mean(dim=1).squeeze(0).cpu())
            raise _EncodingCaptured()

        handle = self.model.model.layers[self.layer].register_forward_hook(capture)
        try:
            try:
                self.model.model(input_ids=ids, use_cache=False)
            except _EncodingCaptured:
                pass
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("SAE capture hook did not execute")
        return captured[0]

    @torch.inference_mode()
    def target_score(self, prompt: str, target: str, *, mean: bool) -> float:
        prefix = self.tokens(prompt)
        tail = self.tokenizer(" " + target, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        if not tail.numel():
            raise ValueError("target must have at least one token")
        ids = torch.cat([prefix, tail], dim=1)
        if ids.shape[1] > self.max_input_tokens:
            raise ValueError("prompt plus target exceeds --max-input-tokens; no silent truncation")
        logits = self.model(input_ids=ids, use_cache=False).logits
        scores = logits[0, prefix.shape[1] - 1:-1].float().log_softmax(-1)
        selected = scores.gather(1, tail[0, :, None]).squeeze(1)
        return float((selected.mean() if mean else selected.sum()).item())

    def label_scores(self, task: str, prompt: str) -> list[float]:
        return [self.target_score(prompt, label, mean=False) for label in LABELS[task]]

    def utility(self, task: str, query: Example, demos: list[Example]) -> float:
        prompt = build_prompt(task, query.input, demos)
        if task in LABELS:
            scores = self.label_scores(task, prompt)
            gold = LABELS[task].index(query.target)
            return scores[gold] - max(s for i, s in enumerate(scores) if i != gold)
        return self.target_score(prompt, query.target, mean=True)

    @torch.inference_mode()
    def predict(self, task: str, prompt: str) -> str:
        if task in LABELS:
            scores = self.label_scores(task, prompt)
            return LABELS[task][max(range(len(scores)), key=scores.__getitem__)]
        ids = self.tokens(prompt)
        output = self.model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
            max_new_tokens=self.max_new_tokens, use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
        return text.split("\n", 1)[0] if task == "commongen" else text.split("\nQuestion:", 1)[0].strip()


def prepare_split(train: list[Example], evaluation: list[Example], args):
    eval_inputs = {normalize_input(e.input) for e in evaluation}
    unique = {}
    removed_overlap = 0
    for row in train:
        key = normalize_input(row.input)
        if key in eval_inputs:
            removed_overlap += 1
        elif key not in unique:
            unique[key] = row
    rows = list(unique.values())
    needed = args.pool_size + args.discovery_queries
    if len(rows) < needed:
        raise ValueError(f"need {needed} distinct train rows after eval-overlap removal; found {len(rows)}")
    picked = random.Random(args.seed).sample(rows, needed)
    return picked[:args.pool_size], picked[args.pool_size:], removed_overlap


def run(args, backend=None):
    started = time.time()
    evaluation = load_examples(args.eval_data, args.task)
    train = load_examples(args.train_data, args.task, require_targets=True)
    if args.task in LABELS:
        if any(row.target not in LABELS[args.task] for row in train + evaluation):
            raise ValueError(f"targets must use canonical labels: {LABELS[args.task]}")
    pool, discovery, removed = prepare_split(train, evaluation, args)
    queries = evaluation[:args.limit] if args.limit else evaluation
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be empty; choose a new run path")
    args.output.mkdir(parents=True, exist_ok=True)
    backend = backend or ModelBackend(args)
    config = RetrievalConfig(n_shot=args.n_shot, shortlist=args.shortlist, beta=args.beta, lambda_r=args.lambda_r)
    print(f"Loaded {len(evaluation)} eval rows; running {len(queries)}; pool={len(pool)}, discovery={len(discovery)}", flush=True)
    utilities, activations = [], []
    for q_index, query in enumerate(discovery):
        q_utilities, q_activations = [], []
        for c_index in range(args.candidate_sets):
            demos = random.Random(args.seed + q_index * 10000 + c_index * 997).sample(pool, args.n_shot)
            if args.task in LABELS:
                demos.sort(key=lambda e: (e.target, e.input))
            q_utilities.append(backend.utility(args.task, query, demos))
            q_activations.append(backend.encode(build_prompt(args.task, query.input, demos)))
        utilities.append(torch.tensor(q_utilities))
        activations.append(torch.stack(q_activations))
        print(f"Discovery {q_index + 1}/{len(discovery)}", flush=True)
    # Query-only utility subtraction cancels in all within-query Eq. 6 pairs.
    weight = learn_pulse_weight(utilities, activations, topk_each_sign=args.topk_each_sign)
    pool_acts = torch.stack([backend.encode(build_prompt(args.task, e.input, [])) for e in pool])
    if not torch.count_nonzero(weight):
        raise ValueError("discovery learned no active features; increase the discovery budget")
    torch.save({"weight": weight, "utilities": utilities, "activations": activations}, args.output / "discovery.pt")
    results = []
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as out:
        for i, query in enumerate(queries):
            query_acts = backend.encode(build_prompt(args.task, query.input, []))
            indices = select_pulse_context(query_acts, pool_acts, weight, config=config)
            prompt = build_prompt(args.task, query.input, [pool[j] for j in indices])
            prediction = backend.predict(args.task, prompt)
            correct = None
            if args.task in LABELS:
                correct = prediction == query.target
            elif args.task == "gsm8k" and query.target is not None:
                expected = numeric_answer(query.target)
                if expected is not None:
                    correct = numeric_answer(prediction) == expected
            row = {"id": query.id, "input": query.input, "reference": query.target,
                   "prediction": prediction, "correct": correct,
                   "selected_ids": [pool[j].id for j in indices],
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            results.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"Predicted {i + 1}/{len(queries)}", flush=True)
    scored = [r for r in results if r["correct"] is not None]
    metrics = {"scored_rows": len(scored), "accuracy": sum(r["correct"] for r in scored) / len(scored) if scored else None}
    if args.task == "commongen":
        metrics["note"] = "Generated text and reference pairs; BLEU is not computed by this runner."
    if args.task == "gsm8k" and not scored:
        metrics["note"] = "No usable reference answers; predictions only."
    summary = {
        "status": "complete", "package_version": __version__, "task": args.task,
        "evaluation_rows_available": len(evaluation), "predicted_rows": len(results),
        "train_rows_available": len(train), "removed_train_eval_overlaps": removed,
        "pool_ids": [e.id for e in pool], "discovery_ids": [e.id for e in discovery],
        "active_features": int(torch.count_nonzero(weight)),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "retrieval": asdict(config), "backend": backend.metadata, "metrics": metrics,
        "eval_sha256": hashlib.sha256(args.eval_data.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256(args.train_data.read_bytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "claim_scope": "Standalone HF integration run; not an independently verified paper-table reproduction.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics), flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=[*LABELS, "commongen", "gsm8k"])
    p.add_argument("--eval-data", type=Path, help="Evaluation TSV; default: bundled dataset/<task>.tsv")
    p.add_argument("--train-data", type=Path, required=True, help="Separate labeled training JSONL or clean task TSV")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    p.add_argument("--sae-id", default="layer_12/width_16k/canonical")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="0: evaluate all rows; positive: first N rows")
    p.add_argument("--pool-size", type=int, default=2000)
    p.add_argument("--discovery-queries", type=int, default=64)
    p.add_argument("--candidate-sets", type=int, default=32)
    p.add_argument("--topk-each-sign", type=int, default=512)
    p.add_argument("--n-shot", type=int, default=4)
    p.add_argument("--shortlist", type=int, default=50)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--lambda-r", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=128)
    args = p.parse_args(argv)
    if args.eval_data is None:
        try:
            args.eval_data = default_eval_data(args.task)
        except FileNotFoundError as exc:
            p.error(str(exc))
    if (min(args.pool_size, args.discovery_queries, args.n_shot, args.topk_each_sign,
            args.max_input_tokens, args.max_new_tokens) < 1
            or args.candidate_sets < 2 or args.limit < 0 or args.n_shot > args.pool_size
            or args.shortlist < args.n_shot
            or args.discovery_queries * args.candidate_sets * (args.candidate_sets - 1) // 2 < 2):
        p.error("invalid sizes: need positive budgets, at least two discovery pairs, and pool/shortlist >= shots")
    RetrievalConfig(n_shot=args.n_shot, shortlist=args.shortlist, beta=args.beta, lambda_r=args.lambda_r)
    try:
        run(args)
    except (ValueError, OSError, ImportError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
