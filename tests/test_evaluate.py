"""Exercise the real discovery/retrieval/output pipeline without model downloads."""

from argparse import Namespace
import hashlib
import json

import pytest
import torch

from pceu_rtr import evaluate
from pceu_rtr.data import Example
from pceu_rtr.evaluate import build_prompt, numeric_answer, run


@pytest.mark.parametrize("task", ["agnews", "rest14", "lap14", "emoc", "commongen", "gsm8k"])
def test_cli_defaults_to_bundled_evaluation_data(task, tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(evaluate, "run", lambda args: captured.append(args))
    evaluate.main(["--task", task, "--train-data", str(tmp_path / "train.jsonl"),
                   "--output", str(tmp_path / "output")])
    assert captured[0].eval_data == evaluate.default_eval_data(task)
    assert captured[0].eval_data.is_file()


def test_cli_keeps_explicit_evaluation_path(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(evaluate, "run", lambda args: captured.append(args))
    custom = tmp_path / "custom.jsonl"
    evaluate.main(["--task", "agnews", "--eval-data", str(custom),
                   "--train-data", str(tmp_path / "train.jsonl"),
                   "--output", str(tmp_path / "output")])
    assert captured[0].eval_data == custom


class FakeBackend:
    metadata = {"kind": "deterministic-test-backend"}

    def __init__(self, prediction="World"):
        self.prediction = prediction
        self.predicted_prompts = []
        self.discovery_calls = []
        self.encoded_prompts = []

    def encode(self, prompt):
        self.encoded_prompts.append(prompt)
        # Candidate contexts vary with their demonstrations, producing real,
        # nonzero discovery covariance and a reproducible selection problem.
        value = int(hashlib.sha256(prompt.encode()).hexdigest()[:6], 16) / 2**24
        return torch.tensor([value, value**2, 1.0 - value, 1.0])

    def utility(self, task, query, demos):
        self.discovery_calls.append((query, list(demos)))
        return float(self.encode(build_prompt(task, query.input, demos))[0])

    def predict(self, task, prompt):
        self.predicted_prompts.append(prompt)
        return self.prediction


def make_args(tmp_path, task="agnews", *, evaluation=None, train=None, limit=0):
    if evaluation is None:
        evaluation = [
            {"id": "eval-0", "input": "Held out query zero", "target": "World"},
            {"id": "eval-1", "input": "Held out query one", "target": "Sports"},
            {"id": "eval-2", "input": "Held out query two", "target": "Business"},
        ]
    if train is None:
        train = [
            {"id": f"train-{i}", "input": f"Independent training example {i}",
             "target": "World" if task == "agnews" else f"Training answer {i} #### {i}"}
            for i in range(12)
        ]
    train_path, eval_path = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    for path, rows in [(train_path, train), (eval_path, evaluation)]:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return Namespace(
        task=task, train_data=train_path, eval_data=eval_path,
        output=tmp_path / "output", seed=42, limit=limit, pool_size=6,
        discovery_queries=3, candidate_sets=4, topk_each_sign=2,
        n_shot=2, shortlist=4, beta=0.3, lambda_r=0.3,
    )


def read_predictions(args):
    return [json.loads(line) for line in (args.output / "predictions.jsonl").read_text().splitlines()]


def test_classification_pipeline_writes_auditable_results(tmp_path):
    args = make_args(tmp_path)
    backend = FakeBackend()
    summary = run(args, backend)
    rows = read_predictions(args)
    assert json.loads((args.output / "summary.json").read_text()) == summary
    assert (args.output / "discovery.pt").is_file()
    assert summary["evaluation_rows_available"] == summary["predicted_rows"] == 3
    assert summary["train_rows_available"] == 12
    assert summary["active_features"] > 0
    assert summary["metrics"] == {"scored_rows": 3, "accuracy": 1 / 3}
    assert [r["reference"] for r in rows] == ["World", "Sports", "Business"]
    assert [r["correct"] for r in rows] == [True, False, False]
    pool, discovery = set(summary["pool_ids"]), set(summary["discovery_ids"])
    assert len(pool) == 6 and len(discovery) == 3 and pool.isdisjoint(discovery)
    assert pool | discovery <= {f"train-{i}" for i in range(12)}
    assert len(backend.discovery_calls) == 12
    for query, demos in backend.discovery_calls:
        assert query.id in discovery
        assert len(demos) == 2 and {e.id for e in demos} <= pool
        assert query.id not in {e.id for e in demos}
    for row, prompt in zip(rows, backend.predicted_prompts):
        assert len(row["selected_ids"]) == len(set(row["selected_ids"])) == 2
        assert set(row["selected_ids"]) <= pool
        assert prompt.endswith(f"Input: {row['input']}\nLabel:")
        assert row["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert summary["eval_sha256"] == hashlib.sha256(args.eval_data.read_bytes()).hexdigest()
    assert summary["train_sha256"] == hashlib.sha256(args.train_data.read_bytes()).hexdigest()


def test_overlap_removed_against_all_eval_rows_even_with_limit(tmp_path):
    args = make_args(tmp_path, limit=1)
    with args.train_data.open("a") as handle:
        for identifier, text in [
            ("overlap-first", "  HELD\tOUT query ZERO  "),
            ("overlap-unprocessed", "held out\nquery TWO"),
        ]:
            handle.write(json.dumps({"id": identifier, "input": text, "target": "World"}) + "\n")
    backend = FakeBackend()
    summary = run(args, backend)
    assert summary["removed_train_eval_overlaps"] == 2
    assert summary["train_rows_available"] == 14
    assert summary["evaluation_rows_available"] == 3
    assert summary["predicted_rows"] == len(read_predictions(args)) == 1
    assert not {"overlap-first", "overlap-unprocessed"} & set(summary["pool_ids"] + summary["discovery_ids"])
    assert all("held out" not in query.input.casefold() for query, _ in backend.discovery_calls)


@pytest.mark.parametrize("task,reference,prediction,correct,scored", [
    ("commongen", "EVAL_REFERENCE_SECRET cat sits on mat", "A cat rests on a mat.", None, 0),
    ("gsm8k", "EVAL_REFERENCE_SECRET rationale #### 1,200", "My computation.\n#### 1200.0", True, 1),
    ("gsm8k", "EVAL_REFERENCE_SECRET rationale #### 1200", "1200 is an intermediate value", False, 1),
    ("gsm8k", "EVAL_REFERENCE_SECRET no numeric answer", "#### 1200", None, 0),
    ("gsm8k", None, "#### 1200", None, 0),
])
def test_generation_reference_handling_without_prompt_leakage(
    tmp_path, task, reference, prediction, correct, scored,
):
    args = make_args(tmp_path, task, evaluation=[
        {"id": "eval-0", "input": "Unique evaluation question", "target": reference},
    ])
    backend = FakeBackend(prediction)
    summary = run(args, backend)
    row = read_predictions(args)[0]
    assert row["reference"] == reference
    assert row["prediction"] == prediction
    assert row["correct"] is correct
    assert summary["metrics"]["scored_rows"] == scored
    assert summary["metrics"]["accuracy"] == (float(correct) if scored else None)
    assert all("EVAL_REFERENCE_SECRET" not in prompt for prompt in backend.encoded_prompts + backend.predicted_prompts)
    assert backend.predicted_prompts[0].endswith("Sentence:" if task == "commongen" else "Answer:")
    if task == "commongen":
        assert "BLEU is not computed" in summary["metrics"]["note"]


def test_train_shortage_after_overlap_and_dedup_rejected_before_inference(tmp_path):
    train = [{"id": f"t-{i}", "input": f"Training {i}", "target": "World"} for i in range(8)]
    train += [
        {"id": "duplicate-input", "input": " TRAINING\t0 ", "target": "World"},
        {"id": "eval-overlap", "input": "Held out query two", "target": "World"},
    ]
    args = make_args(tmp_path, train=train, limit=1)
    backend = FakeBackend()
    with pytest.raises(ValueError, match="need 9 distinct train rows.*found 8"):
        run(args, backend)
    assert not backend.encoded_prompts and not backend.predicted_prompts
    assert not args.output.exists()


def test_existing_results_are_not_overwritten(tmp_path):
    args = make_args(tmp_path)
    args.output.mkdir()
    sentinel = args.output / "predictions.jsonl"
    sentinel.write_text("previous results\n")
    with pytest.raises(ValueError, match="output directory must be empty"):
        run(args, FakeBackend())
    assert sentinel.read_text() == "previous results\n"


def test_prompt_contains_demonstration_target_only():
    prompt = build_prompt("commongen", "eval concepts", [Example("demo", "train concepts", "demo answer")])
    assert "Sentence: demo answer\n\nConcepts: eval concepts\nSentence:" in prompt
    with pytest.raises(ValueError, match="demonstrations require targets"):
        build_prompt("gsm8k", "query", [Example("demo", "question", None)])


@pytest.mark.parametrize("text,expected", [
    ("Reasoning 99, then #### 1,200.00", "1.2E+3"),
    ("#### -12.50", "-12.5"),
    ("#### +42", "42"),
    ("#### 0.125", "0.125"),
    ("The answer is 42", None),
    ("Reasoning 42\n#### unknown", None),
    ("", None),
])
def test_numeric_answer_requires_declared_final_answer(text, expected):
    assert numeric_answer(text) == expected
