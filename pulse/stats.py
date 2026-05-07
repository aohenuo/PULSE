from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class OnlineVectorStats:
    count: int
    mean: torch.Tensor
    m2: torch.Tensor

    @classmethod
    def create(cls, dim: int, device: torch.device):
        return cls(count=0, mean=torch.zeros(dim, device=device), m2=torch.zeros(dim, device=device))

    def update(self, x: torch.Tensor):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def variance(self) -> torch.Tensor:
        if self.count < 2:
            return torch.zeros_like(self.mean)
        return self.m2 / (self.count - 1)


# ---------------------------------------------------------------------------
# Statistical testing utilities (stdlib-only, no numpy)
# ---------------------------------------------------------------------------

import math
import random
from typing import List, Tuple


def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean of *values*.

    Returns (mean, ci_lower, ci_upper).
    """
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty")

    rng = random.Random(seed)
    observed_mean = sum(values) / n

    boot_means: List[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = 1.0 - confidence
    lo_idx = int(math.floor((alpha / 2) * n_bootstrap))
    hi_idx = int(math.floor((1.0 - alpha / 2) * n_bootstrap)) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))

    return (observed_mean, boot_means[lo_idx], boot_means[hi_idx])


def paired_bootstrap_test(
    values_a: List[float],
    values_b: List[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Paired bootstrap test for the difference in means (A - B).

    Tests H0: mean(A) == mean(B).
    Returns (mean_diff, ci_lower, ci_upper, p_value).
    """
    n = len(values_a)
    if n != len(values_b):
        raise ValueError("values_a and values_b must have the same length")
    if n == 0:
        raise ValueError("inputs must be non-empty")

    diffs = [a - b for a, b in zip(values_a, values_b)]
    observed_diff = sum(diffs) / n

    rng = random.Random(seed)
    boot_diffs: List[float] = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randint(0, n - 1)] for _ in range(n)]
        boot_diffs.append(sum(sample) / n)

    boot_diffs.sort()
    lo_idx = int(math.floor(0.025 * n_bootstrap))
    hi_idx = int(math.floor(0.975 * n_bootstrap)) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))

    # Two-sided p-value: proportion of bootstrap diffs on the opposite side of zero
    count_extreme = sum(1 for d in boot_diffs if d * observed_diff <= 0)
    p_value = count_extreme / n_bootstrap if n_bootstrap > 0 else 1.0

    return (observed_diff, boot_diffs[lo_idx], boot_diffs[hi_idx], p_value)


def mcnemar_test(
    correct_a: List[int],
    correct_b: List[int],
) -> Tuple[float, float]:
    """McNemar's test for paired binary outcomes.

    *correct_a* and *correct_b* are lists of 0/1 indicating whether each
    item was classified correctly by system A and B respectively.

    Uses the chi-squared approximation:
        chi2 = (b - c)^2 / (b + c)
    where b = count(A wrong & B right), c = count(A right & B wrong).

    Returns (chi2, p_value).
    """
    n = len(correct_a)
    if n != len(correct_b):
        raise ValueError("correct_a and correct_b must have the same length")

    # b = A wrong, B right;  c = A right, B wrong
    b = sum(1 for a, bv in zip(correct_a, correct_b) if a == 0 and bv == 1)
    c = sum(1 for a, bv in zip(correct_a, correct_b) if a == 1 and bv == 0)

    if b + c == 0:
        return (0.0, 1.0)

    chi2 = (b - c) ** 2 / (b + c)

    # Survival function of chi-squared(df=1) via the normal approximation:
    # P(X > x) = erfc(sqrt(x/2)) / 2  for df=1 chi-squared
    p_value = math.erfc(math.sqrt(chi2 / 2)) / 2 if chi2 > 0 else 1.0

    return (chi2, p_value)


def wilson_ci(
    n_success: int,
    n_total: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Wilson score confidence interval for a proportion.

    Returns (center, lower, upper).
    """
    if n_total == 0:
        raise ValueError("n_total must be > 0")

    p_hat = n_success / n_total
    # z-score for the confidence level (two-sided)
    # For 0.95 -> z ~ 1.96, 0.99 -> z ~ 2.576
    # Use inverse of normal CDF via erfinv approximation
    alpha = 1.0 - confidence
    z = _normal_ppf(1.0 - alpha / 2)

    denom = 1 + z * z / n_total
    center = (p_hat + z * z / (2 * n_total)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z * z / (4 * n_total * n_total)) / denom

    return (center, center - margin, center + margin)


def _normal_ppf(p: float) -> float:
    """Approximate inverse of the standard normal CDF (percent-point function).

    Uses the rational approximation from Abramowitz & Stegun (formula 26.2.23)
    which is accurate to ~4.5e-4.  Sufficient for common confidence levels.
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -_normal_ppf(1.0 - p)

    # Coefficients for rational approximation
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
