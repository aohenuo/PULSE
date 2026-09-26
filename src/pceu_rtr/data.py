"""Load source examples without depending on prediction or analysis columns."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Example:
    id: str
    input: str
    target: str | None


_TASK_COLUMNS = {
    "agnews": ("text", "gold"),
    "rest14": ("text", "gold"),
    "lap14": ("text", "gold"),
    "emoc": ("text", "gold"),
    "commongen": ("concepts", "reference"),
    "gsm8k": ("question", "answer"),
}


def normalize_input(value: str) -> str:
    """Normalize only for overlap checks; loaded examples retain original text."""
    return " ".join(value.split()).casefold()


def load_examples(
    path: str | Path, task: str, *, require_targets: bool = False,
) -> list[Example]:
    """Read task-specific TSV or standard ``id,input,target`` JSONL.

    Only GSM8K evaluation may omit targets. Pass ``require_targets=True`` for
    training/discovery data. Extra columns are ignored, never used as labels.
    Errors include the source file and row/line to make bad inputs actionable.
    """
    path = Path(path)
    if task not in _TASK_COLUMNS:
        raise ValueError(f"Unknown task {task!r}; expected one of {', '.join(_TASK_COLUMNS)}")
    if path.suffix.lower() not in {".tsv", ".jsonl"}:
        raise ValueError(f"{path}: expected a .tsv or .jsonl file")
    needs_target = require_targets or task != "gsm8k"
    examples: list[Example] = []
    seen: set[str] = set()

    def add(identifier: object, text: object, target: object, line: int) -> None:
        where = f"{path}:{line}"
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
            raise ValueError(f"{where}: id must be a nonempty string or integer")
        identifier = str(identifier)
        if not identifier.strip():
            raise ValueError(f"{where}: id must be nonempty")
        if identifier in seen:
            raise ValueError(f"{where}: duplicate id {identifier!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{where}: input must be a nonempty string")
        if target is not None and (not isinstance(target, str) or not target.strip()):
            raise ValueError(f"{where}: target must be a nonempty string or null")
        if needs_target and target is None:
            raise ValueError(f"{where}: target is required for {task}")
        seen.add(identifier)
        examples.append(Example(identifier, text, target))

    with path.open(encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() == ".jsonl":
            for line, raw in enumerate(handle, 1):
                if not raw.strip():
                    raise ValueError(f"{path}:{line}: empty JSONL record")
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line}: malformed JSON: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line}: JSONL record must be an object")
                add(record.get("id"), record.get("input"), record.get("target"), line)
        else:
            reader = csv.DictReader(handle, delimiter="\t", strict=True)
            try:
                headers = reader.fieldnames
                if not headers:
                    raise ValueError(f"{path}: missing TSV header")
                if len(headers) != len(set(headers)) or any(not h.strip() for h in headers):
                    raise ValueError(f"{path}: duplicate or empty TSV column name")
                input_column, target_column = _TASK_COLUMNS[task]
                if task == "gsm8k":
                    if "answer" in headers and "target" in headers:
                        raise ValueError(f"{path}: use one GSM8K target column: answer or target")
                    target_column = "answer" if "answer" in headers else "target"
                required = {"index", input_column}
                if needs_target:
                    required.add(target_column)
                missing = required.difference(headers)
                if missing:
                    raise ValueError(f"{path}: missing TSV columns: {', '.join(sorted(missing))}")
                for record in reader:
                    if None in record or any(value is None for value in record.values()):
                        raise ValueError(f"{path}:{reader.line_num}: row has wrong number of TSV fields")
                    add(record["index"], record[input_column], record.get(target_column), reader.line_num)
            except csv.Error as exc:
                raise ValueError(f"{path}:{reader.line_num}: malformed TSV: {exc}") from exc
    if not examples:
        raise ValueError(f"{path}: dataset contains no examples")
    return examples
