"""Frozen pools remain comparable even when they exhaust clean training data."""
from argparse import Namespace
import hashlib
import json
import random

import pytest

from pceu_rtr.data import Example
from pceu_rtr.splits import prepare_frozen_split


def setup_split(tmp_path):
    train = [Example(str(i), f"Train input {i}", "World") for i in range(6)]
    train += [Example("duplicate", "  TRAIN input 0  ", "World"),
              Example("overlap", " held OUT ", "World")]
    evaluation = [Example("eval", "Held out", "World")]
    train_path, eval_path = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    for path, rows in [(train_path, train), (eval_path, evaluation)]:
        path.write_text("".join(json.dumps(vars(row)) + "\n" for row in rows))
    summary = {
        "status": "complete", "task": "agnews",
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "eval_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
        "pool_ids": ["3", "5", "1", "0", "4", "2"],
    }
    source = tmp_path / "baseline.json"
    source.write_text(json.dumps(summary))
    args = Namespace(pool_from_summary=source, task="agnews", train_data=train_path,
                     eval_data=eval_path, pool_size=6, discovery_queries=3, seed=42, n_shot=4)
    return train, evaluation, args, summary


def test_exhausted_pool_preserves_order_and_independent_training_discovery(tmp_path):
    train, evaluation, args, source = setup_split(tmp_path)
    pool, discovery, removed, metadata = prepare_frozen_split(train, evaluation, args)
    expected = train[:6].copy()
    random.Random(args.seed).shuffle(expected)
    assert [row.id for row in pool] == source["pool_ids"]
    assert discovery == expected[:3]
    assert removed == 1
    assert metadata["clean_training_rows"] == 6
    assert metadata["discovery_pool_overlap"] == 3
    assert metadata["pool_order_preserved"] is True
    assert metadata["source_summary_sha256"] == hashlib.sha256(args.pool_from_summary.read_bytes()).hexdigest()
    assert all(row.id not in {"eval", "overlap", "duplicate"} for row in discovery)
    again = prepare_frozen_split(train, evaluation, args)
    assert again == (pool, discovery, removed, metadata)


@pytest.mark.parametrize("change,match", [
    ({"status": "running"}, "complete"),
    ({"task": "lap14"}, "task"),
    ({"train_sha256": "bad"}, "train SHA256"),
    ({"eval_sha256": "bad"}, "eval SHA256"),
    ({"pool_ids": ["0"] * 6}, "duplicate"),
    ({"pool_ids": ["0", "1"]}, "count"),
    ({"pool_ids": [0, "1", "2", "3", "4", "5"]}, "string pool IDs"),
    ({"pool_ids": ["missing", "1", "2", "3", "4", "5"]}, "absent"),
    ({"pool_ids": ["duplicate", "1", "2", "3", "4", "5"]}, "absent"),
    ({"pool_ids": ["overlap", "1", "2", "3", "4", "5"]}, "absent"),
])
def test_rejects_invalid_source(tmp_path, change, match):
    train, evaluation, args, source = setup_split(tmp_path)
    source.update(change)
    args.pool_from_summary.write_text(json.dumps(source))
    with pytest.raises(ValueError, match=match):
        prepare_frozen_split(train, evaluation, args)


@pytest.mark.parametrize("count", [0, -1, 7])
def test_rejects_invalid_discovery_budget(tmp_path, count):
    train, evaluation, args, _ = setup_split(tmp_path)
    args.discovery_queries = count
    with pytest.raises(ValueError, match="discovery_queries"):
        prepare_frozen_split(train, evaluation, args)


def test_requires_room_for_self_exclusion(tmp_path):
    train, evaluation, args, _ = setup_split(tmp_path)
    args.n_shot = 6
    with pytest.raises(ValueError, match="non-self"):
        prepare_frozen_split(train, evaluation, args)


def test_changed_source_order_is_followed_exactly(tmp_path):
    train, evaluation, args, source = setup_split(tmp_path)
    source["pool_ids"].reverse()
    args.pool_from_summary.write_text(json.dumps(source))
    pool, _, _, _ = prepare_frozen_split(train, evaluation, args)
    assert [row.id for row in pool] == source["pool_ids"]
