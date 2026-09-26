"""Bind supplemental evaluations to an existing, auditable support pool."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from .data import Example, normalize_input


def prepare_frozen_split(train: list[Example], evaluation: list[Example], args):
    """Reuse a completed run's ordered pool and sample training-only discovery.

    Discovery may overlap the support pool. Callers must exclude the current
    discovery input from its demonstration candidates, as in paper section 3.2.
    """
    source_path = Path(args.pool_from_summary)
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    if not isinstance(source, dict) or source.get("status") != "complete":
        raise ValueError("frozen pool source must be a complete summary")
    if source.get("task") != args.task:
        raise ValueError("frozen pool source task does not match")
    for name in ("train", "eval"):
        actual = hashlib.sha256(Path(getattr(args, f"{name}_data")).read_bytes()).hexdigest()
        if source.get(f"{name}_sha256") != actual:
            raise ValueError(f"frozen pool source {name} SHA256 does not match")

    pool_ids = source.get("pool_ids")
    if (not isinstance(pool_ids, list) or not pool_ids
            or any(not isinstance(identifier, str) or not identifier.strip() for identifier in pool_ids)):
        raise ValueError("frozen pool source requires nonempty string pool IDs")
    if len(set(pool_ids)) != len(pool_ids):
        raise ValueError("frozen pool source has duplicate pool IDs")
    if len(pool_ids) != args.pool_size:
        raise ValueError("frozen pool source count does not match pool_size")

    eval_inputs = {normalize_input(row.input) for row in evaluation}
    unique = {}
    removed_overlap = 0
    for row in train:
        key = normalize_input(row.input)
        if key in eval_inputs:
            removed_overlap += 1
        elif key not in unique:
            unique[key] = row
    clean_rows = list(unique.values())
    by_id = {row.id: row for row in clean_rows}
    if any(identifier not in by_id for identifier in pool_ids):
        raise ValueError("frozen pool ID is absent from deduplicated evaluation-disjoint training rows")
    pool = [by_id[identifier] for identifier in pool_ids]
    if args.discovery_queries < 1 or args.discovery_queries > len(clean_rows):
        raise ValueError("discovery_queries must be positive and no larger than clean training rows")
    discovery_rows = list(clean_rows)
    random.Random(args.seed).shuffle(discovery_rows)
    discovery = discovery_rows[:args.discovery_queries]
    pool_inputs = {normalize_input(row.input) for row in pool}
    overlap = sum(normalize_input(row.input) in pool_inputs for row in discovery)
    if args.n_shot < 1 or len(pool) - int(overlap > 0) < args.n_shot:
        raise ValueError("frozen pool lacks sufficient non-self demonstration candidates")
    metadata = {
        "policy": "frozen_ordered_pool_independent_training_discovery",
        "source_summary": str(source_path),
        "source_summary_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "discovery_sampling": "random.Random(seed).shuffle(clean_training_rows); first discovery_queries",
        "discovery_pool_overlap": overlap,
        "discovery_self_exclusion": "normalized_input",
        "clean_training_rows": len(clean_rows),
        "pool_order_preserved": True,
    }
    return pool, discovery, removed_overlap, metadata
