from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pceu_rtr import (
    RetrievalConfig,
    classification_fusion_relevance,
    full_sae_greedy_select,
    generation_fusion_relevance,
    learn_pulse_weight,
    pair_feature_statistics,
    pulse_eq9_relevance,
    select_classification_context,
    select_generation_context,
    standardize_scores,
)


def test_pair_statistics_match_explicit_all_pairs() -> None:
    utilities = torch.tensor([0.2, 1.1, -0.4, 0.7])
    activations = torch.tensor(
        [[0.0, 1.0], [2.0, -1.0], [1.0, 0.5], [-0.5, 3.0]]
    )
    observed = pair_feature_statistics(utilities, activations)
    deltas_u = []
    deltas_a = []
    for left in range(utilities.numel()):
        for right in range(left + 1, utilities.numel()):
            deltas_u.append(utilities[left] - utilities[right])
            deltas_a.append(activations[left] - activations[right])
    delta_u = torch.stack(deltas_u).double()
    delta_a = torch.stack(deltas_a).double()
    assert observed.pair_count == 6
    assert torch.allclose(observed.numerator, (delta_u[:, None] * delta_a).sum(0))
    assert torch.allclose(observed.delta_sum, delta_a.sum(0))
    assert torch.allclose(observed.delta_square_sum, delta_a.square().sum(0))


def test_learn_weight_is_sparse_signed_and_deterministic() -> None:
    utilities = [torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0, -1.0, 0.5])]
    activations = [
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.5, 0.0], [2.0, -1.0, -1.0]]),
        torch.tensor([[1.0, -1.0, 0.0], [-1.0, 2.0, 1.0], [0.5, 0.0, -0.5]]),
    ]
    first = learn_pulse_weight(utilities, activations, topk_each_sign=1)
    second = learn_pulse_weight(utilities, activations, topk_each_sign=1)
    assert torch.equal(first, second)
    assert int((first > 0).sum()) <= 1
    assert int((first < 0).sum()) <= 1


def test_standardize_is_affine_invariant_and_constant_safe() -> None:
    values = torch.tensor([-2.0, 0.0, 1.0, 4.0])
    assert torch.allclose(
        standardize_scores(7.0 * values + 19.0), standardize_scores(values)
    )
    assert torch.equal(standardize_scores(torch.ones(4)), torch.zeros(4))


def test_eq9_uses_weight_magnitude_not_sign() -> None:
    generator = torch.Generator().manual_seed(3)
    query = torch.rand(6, generator=generator)
    pool = torch.rand((9, 6), generator=generator)
    weight = torch.tensor([2.0, -1.0, 0.0, 0.5, 0.0, -0.25])
    positive = pulse_eq9_relevance(query, pool, weight)
    flipped = pulse_eq9_relevance(query, pool, -weight)
    assert torch.allclose(positive, flipped)
    assert positive.shape == (9,)


def test_classification_fusion_matches_validated_formula() -> None:
    pulse = torch.tensor([0.0, 1.0, 4.0, 9.0])
    semantic = torch.tensor([9.0, 4.0, 1.0, 0.0])
    expected = standardize_scores(
        0.5 * standardize_scores(pulse) + 0.5 * standardize_scores(semantic)
    )
    assert torch.allclose(
        classification_fusion_relevance(pulse, semantic), expected
    )
    assert torch.allclose(
        classification_fusion_relevance(pulse, semantic, alpha=1.0),
        standardize_scores(pulse),
    )


def test_generation_fusion_restores_eq9_scale() -> None:
    pulse = torch.tensor([-0.8, -0.1, 0.2, 0.5, 1.4])
    semantic = torch.tensor([0.9, 0.2, -0.4, 0.3, -0.1])
    fused = generation_fusion_relevance(pulse, semantic)
    assert float(fused.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(fused.std(unbiased=True)) == pytest.approx(
        float(pulse.std(unbiased=True)), abs=1e-6
    )


def test_full_sae_greedy_avoids_a_near_duplicate() -> None:
    relevance = torch.tensor([3.0, 2.9, 2.8, 0.0])
    pool = torch.tensor([[1.0, 0.0], [0.999, 0.01], [0.0, 1.0], [-1.0, 0.0]])
    selected = full_sae_greedy_select(
        relevance, pool, n_shot=2, shortlist=4, lambda_r=1.0
    )
    assert selected == [0, 2]


def test_high_level_selectors_return_deterministic_four_shot_contexts() -> None:
    generator = torch.Generator().manual_seed(17)
    pool_sae = torch.rand((12, 7), generator=generator)
    query_sae = torch.rand(7, generator=generator)
    pool_sbert = torch.rand((12, 5), generator=generator)
    query_sbert = torch.rand(5, generator=generator)
    weight = torch.tensor([2.0, -1.0, 0.0, 0.5, 0.0, -0.25, 0.75])
    normalized = F.normalize(pool_sae.float(), dim=1)
    config = RetrievalConfig(shortlist=8)

    classification = select_classification_context(
        query_sae,
        query_sbert,
        pool_sae,
        pool_sbert,
        weight,
        config=config,
        pool_sae_normalized=normalized,
    )
    repeated = select_classification_context(
        query_sae,
        query_sbert,
        pool_sae,
        pool_sbert,
        weight,
        config=config,
        pool_sae_normalized=normalized,
    )
    generation = select_generation_context(
        query_sae,
        query_sbert,
        pool_sae,
        pool_sbert,
        weight,
        config=config,
        pool_sae_normalized=normalized,
    )
    assert classification == repeated
    assert len(classification) == len(set(classification)) == 4
    assert len(generation) == len(set(generation)) == 4


def test_config_rejects_invalid_formal_parameters() -> None:
    with pytest.raises(ValueError, match="alpha"):
        RetrievalConfig(alpha=1.1)
    with pytest.raises(ValueError, match="lambda_r"):
        RetrievalConfig(lambda_r=-0.1)
