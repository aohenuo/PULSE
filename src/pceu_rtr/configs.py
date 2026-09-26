"""Defaults retained from the PCEU/PULSE research implementation."""
from __future__ import annotations

from dataclasses import dataclass


FORMAL_DISCOVERY_QUERIES = 64
FORMAL_CANDIDATE_SETS_PER_QUERY = 32
FORMAL_TOPK_EACH_SIGN = {
    "gemma": 512,
    "llama": 2_048,
}


@dataclass(frozen=True)
class RetrievalConfig:
    """Settings for SAE retrieval and optional semantic fusion."""

    alpha: float = 0.5
    beta: float = 0.3
    n_shot: int = 4
    shortlist: int = 50
    lambda_r: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if self.n_shot < 0 or self.shortlist < 0:
            raise ValueError("n_shot and shortlist must be non-negative")
        if self.lambda_r < 0:
            raise ValueError("lambda_r must be non-negative")


DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()


