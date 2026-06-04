"""Metric helpers used by the public aggregate tables."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable


def rate(successes: int, total: int) -> float:
    if total <= 0:
        raise ValueError("total must be positive")
    return successes / total


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial rate."""

    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) / total) + (z * z / (4 * total * total))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def cohen_kappa(pairs: Iterable[tuple[int, int]]) -> tuple[float, float]:
    """Return observed agreement and Cohen's kappa for binary labels."""

    pairs = list(pairs)
    if not pairs:
        raise ValueError("pairs must be non-empty")
    total = len(pairs)
    observed = sum(1 for left, right in pairs if int(left) == int(right)) / total
    left_counts = Counter(int(left) for left, _ in pairs)
    right_counts = Counter(int(right) for _, right in pairs)
    expected = sum((left_counts[label] / total) * (right_counts[label] / total) for label in (0, 1))
    if math.isclose(1.0, expected):
        return observed, 1.0
    return observed, (observed - expected) / (1 - expected)
