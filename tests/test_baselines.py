import random

import pytest
import torch

from pceu_rtr.baselines import (
    BASELINE_METHODS, BASELINE_METADATA, greedy_map_dpp, jaccard_similarity,
    lexical_token_set, select_baseline, static_sbert_dpp_kernel,
)


def test_pure_sbert_is_scale_invariant_topk_with_stable_ties():
    selected = select_baseline("sbert", pool_texts=["a", "b", "c"], query_text="x",
        query_sbert=torch.tensor([2., 0.]), pool_sbert=torch.tensor([[10., 0.], [1., 0.], [0., 2.]]),
        n_shot=2, shortlist=1)
    assert selected == [0, 1]  # No SAE redundancy and no shortlist restriction.


def test_knn_sae_retains_matched_redundancy_composer():
    selected = select_baseline("knn_sae", pool_texts=["a", "b", "c", "d"], query_text="x",
        query_sae=torch.tensor([1., 0.]), pool_sae=torch.tensor([[1., 0.], [1., 0.], [.9, .43589], [0., 1.]]),
        n_shot=2, lambda_r=3.)
    assert selected == [0, 3]


def test_lexical_punctuation_and_matched_composition():
    assert jaccard_similarity(lexical_token_set("Great food, GREAT"), lexical_token_set("great food")) == pytest.approx(1 / 3)
    assert jaccard_similarity(set(), set()) == 0
    assert select_baseline("lex_sim", pool_texts=["a", "a", "a b"], query_text="a",
        pool_sae=torch.tensor([[1., 0.], [1., 0.], [0., 1.]]), n_shot=2, lambda_r=1.) == [0, 2]


def test_ceil_diversifies_and_is_explicitly_a_reconstruction():
    query = torch.tensor([1., 0.])
    candidates = torch.tensor([[1., 0.], [.999, .001], [0., 1.]])
    kernel = static_sbert_dpp_kernel(query, candidates)
    assert torch.allclose(kernel, kernel.T)
    assert greedy_map_dpp(kernel, 2) == [0, 2]
    assert select_baseline("ceil", pool_texts=["a", "b", "c"], query_text="x",
        query_sbert=query, pool_sbert=candidates, n_shot=2) == [0, 2]
    assert BASELINE_METADATA["ceil"]["official_checkpoint"] is False


def test_random_matches_source_seed_rule_without_global_rng_mutation():
    state = random.getstate()
    indices = select_baseline("random", pool_texts=[str(i) for i in range(20)], query_text="x", query_index=3)
    assert indices == random.Random(42 + 7919 * 3).sample(range(20), 4)
    assert random.getstate() == state
    assert select_baseline("random", pool_texts=["a"], query_text="x") == [0]


@pytest.mark.parametrize("method", BASELINE_METHODS)
def test_empty_pool_and_zero_budget(method):
    common = dict(pool_texts=[], query_text="x", pool_sae=torch.empty(0, 2), pool_sbert=torch.empty(0, 2),
                  query_sae=torch.zeros(2), query_sbert=torch.zeros(2))
    assert select_baseline(method, **common) == []
    common.update(pool_texts=["x"], pool_sae=torch.zeros(1, 2), pool_sbert=torch.zeros(1, 2))
    assert select_baseline(method, n_shot=0, **common) == []


@pytest.mark.parametrize("kwargs", [{"n_shot": -1}, {"n_shot": 1.5}, {"shortlist": -1},
    {"query_index": -1}, {"lambda_r": float("nan")}, {"lambda_r": float("inf")}])
def test_invalid_budgets_fail(kwargs):
    with pytest.raises(ValueError):
        select_baseline("random", pool_texts=["a"], query_text="x", **kwargs)


def test_missing_mismatched_and_nonfinite_features_fail():
    with pytest.raises(ValueError, match="pool_sbert"):
        select_baseline("sbert", pool_texts=["a"], query_text="x")
    with pytest.raises(ValueError, match="dimension"):
        select_baseline("knn_sae", pool_texts=["a"], query_text="x", pool_sae=torch.ones(1, 2), query_sae=torch.ones(3))
    with pytest.raises(ValueError, match="finite"):
        select_baseline("lex_sim", pool_texts=["a"], query_text="x", pool_sae=torch.tensor([[float("nan")]]))
    with pytest.raises(ValueError, match="unknown baseline"):
        select_baseline("epr", pool_texts=["a"], query_text="x")
