"""Paired comparison of two retrieval arms scored on the same queries.

With a few hundred queries, a gap of a point or two between two systems is often noise. These
helpers give the paired mean difference with a bootstrap confidence interval, and win/loss counts.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedComparison:
    n: int
    mean_difference: float  # mean of (b - a): positive means b is better
    ci_low: float
    ci_high: float
    b_better: int
    a_better: int
    ties: int

    @property
    def significant(self) -> bool:
        """True if the 95% interval excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0


def compare_paired(
    a: Sequence[float],
    b: Sequence[float],
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Compare per-query scores ``a`` and ``b`` (same queries, same order), paired bootstrap."""
    if len(a) != len(b):
        raise ValueError("a and b must score the same queries")
    if not a:
        raise ValueError("no queries to compare")
    if not 0 < confidence < 1 or resamples < 1:
        raise ValueError("confidence must be in (0, 1) and resamples >= 1")
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    tail = (1 - confidence) / 2
    low = means[int(tail * resamples)]
    high = means[min(resamples - 1, int((1 - tail) * resamples))]
    eps = 1e-12
    return PairedComparison(
        n=n,
        mean_difference=sum(diffs) / n,
        ci_low=low,
        ci_high=high,
        b_better=sum(d > eps for d in diffs),
        a_better=sum(d < -eps for d in diffs),
        ties=sum(abs(d) <= eps for d in diffs),
    )
