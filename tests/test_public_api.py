import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from pceu_rtr import (
    RetrievalConfig, full_sae_greedy_select, learn_pulse_weight,
    pulse_eq9_relevance, rank_candidate_sets, select_pulse_context,
)
from pceu_rtr.cli import select_from_payload


def payload():
    return {
        "discovery": [{"utilities": [0., 0.5, 1.],
                       "activations": [[0., 2.], [1., 1.], [2., 0.]]}],
        "topk_each_sign": 1, "query_sae": [2., 0.],
        "pool_sae": [[2., 0.], [0., 2.], [1., 1.]],
        "pool_ids": ["a", "b", "c"],
    }


def test_discovery_matches_explicit_global_pairs():
    utilities = [torch.tensor([0., 1., 3.]), torch.tensor([1., -2., 2., 5.])]
    acts = [torch.tensor([[0., 3.], [1., 1.], [3., 0.]]),
            torch.tensor([[1., 2.], [0., 4.], [2., 1.], [4., 0.]])]
    du, da = [], []
    for u, a in zip(utilities, acts):
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                du.append((u[i] - u[j]).double())
                da.append((a[i] - a[j]).double())
    du, da = torch.stack(du), torch.stack(da)
    expected = (du[:, None] * da).mean(0) / (da.var(0, unbiased=True) + 1e-6).sqrt()
    actual = learn_pulse_weight(utilities, acts, topk_each_sign=2)
    assert torch.allclose(actual, expected.float())


def test_signed_complete_context_ranking():
    delta = torch.tensor([[2., 3.], [1., 0.], [0., -1.]])
    scores, order = rank_candidate_sets(delta, torch.tensor([1., -1.]))
    assert scores.tolist() == [-1., 1., 1.]
    assert order == [1, 2, 0]
    assert rank_candidate_sets(delta, torch.tensor([-1., 1.]))[1] == [0, 1, 2]


def test_default_selects_paper_relevance_without_outer_normalization():
    q = torch.tensor([2., 1., 0.])
    pool = torch.tensor([[1., 0., 0.], [0., 2., 1.], [2., 1., 0.]])
    weight = torch.tensor([2., -1., 0.])
    config = RetrievalConfig(n_shot=2)
    expected = full_sae_greedy_select(
        pulse_eq9_relevance(q, pool, weight), pool, n_shot=2,
    )
    assert select_pulse_context(q, pool, weight, config=config) == expected


def test_cli_outputs_selected_ids_and_fails_cleanly(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload()))
    result = subprocess.run(
        [sys.executable, "-m", "pceu_rtr.cli", str(path), "--n-shot", "2"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    output = json.loads(result.stdout)
    assert output["indices"] == [0, 2]
    assert output["selected_ids"] == ["a", "c"]
    assert output["method"] == "pulse"
    assert len(output["input_sha256"]) == 64
    path.write_text('{"query_sae": [1]}')
    failed = subprocess.run(
        [sys.executable, "-m", "pceu_rtr.cli", str(path)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert failed.returncode == 2
    assert "error:" in failed.stderr
    assert "Traceback" not in failed.stderr


def test_json_weight_replay_and_duplicate_ids():
    data = payload()
    first = select_from_payload(data, RetrievalConfig(n_shot=2), "pulse")
    data.pop("discovery")
    data["pulse_weight"] = first["pulse_weight"]
    assert select_from_payload(data, RetrievalConfig(n_shot=2), "pulse")["indices"] == first["indices"]
    data["pool_ids"] = ["a", "a", "c"]
    with pytest.raises(ValueError, match="unique"):
        select_from_payload(data, RetrievalConfig(), "pulse")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_discovery_is_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        learn_pulse_weight([torch.tensor([0., value, 1.])],
                           [torch.ones(3, 2)], topk_each_sign=1)


def test_zero_weight_and_empty_selection():
    data = payload()
    data.pop("discovery")
    data["pulse_weight"] = [0., 0.]
    with pytest.raises(ValueError, match="active features"):
        select_from_payload(data, RetrievalConfig(), "pulse")
    assert full_sae_greedy_select(torch.ones(3), torch.ones(3, 2), n_shot=0) == []
    assert len(full_sae_greedy_select(torch.ones(3), torch.eye(3), n_shot=8)) == 3


@pytest.mark.parametrize("data", [[], {"discovery": None}, {"discovery": [None]}])
def test_malformed_json_structure_is_rejected(data):
    with pytest.raises(ValueError):
        select_from_payload(data, RetrievalConfig(), "pulse")
