"""PULSE tensor core and explicitly named semantic fusion extensions."""

from .configs import (
    DEFAULT_RETRIEVAL_CONFIG,
    FORMAL_CANDIDATE_SETS_PER_QUERY,
    FORMAL_DISCOVERY_QUERIES,
    FORMAL_TOPK_EACH_SIGN,
    RetrievalConfig,
)
from .core import (
    PairFeatureStatistics,
    classification_fusion_relevance,
    feature_scores_from_statistics,
    full_sae_greedy_select,
    generation_fusion_relevance,
    learn_pulse_weight,
    merge_pair_feature_statistics,
    pair_feature_statistics,
    pulse_eq9_relevance,
    rank_candidate_sets,
    select_classification_context,
    select_generation_context,
    select_pulse_context,
    semantic_relevance,
    sparse_signed_topk,
    standardize_scores,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_RETRIEVAL_CONFIG", "FORMAL_CANDIDATE_SETS_PER_QUERY",
    "FORMAL_DISCOVERY_QUERIES", "FORMAL_TOPK_EACH_SIGN", "RetrievalConfig",
    "PairFeatureStatistics", "classification_fusion_relevance",
    "feature_scores_from_statistics", "full_sae_greedy_select",
    "generation_fusion_relevance", "learn_pulse_weight",
    "merge_pair_feature_statistics", "pair_feature_statistics",
    "pulse_eq9_relevance", "rank_candidate_sets", "select_classification_context",
    "select_generation_context", "select_pulse_context", "semantic_relevance",
    "sparse_signed_topk", "standardize_scores",
]
