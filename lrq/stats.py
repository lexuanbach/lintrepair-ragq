"""Statistics helpers: Wilson score interval and paired bootstrap.

Per prior lessons, when methods are evaluated on the SAME items the powered
comparison is the *paired* difference (resample the item, recompute the
within-resample method-minus-baseline), not overlap of marginal CIs.
"""
from __future__ import annotations

import math
import random


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (point, lo, hi) Wilson score interval for k successes of n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def paired_bootstrap_diff(a: list[int], b: list[int], iters: int = 10000,
                          seed: int = 12345) -> tuple[float, float, float]:
    """Paired bootstrap of mean(a) - mean(b) over matched items a[i], b[i].

    Returns (point_diff, lo, hi) for a 95% percentile CI.
    """
    assert len(a) == len(b) and a
    rng = random.Random(seed)
    n = len(a)
    point = sum(a) / n - sum(b) / n
    diffs = []
    idx = range(n)
    for _ in range(iters):
        s = [rng.randrange(n) for _ in idx]
        da = sum(a[i] for i in s) / n
        db = sum(b[i] for i in s) / n
        diffs.append(da - db)
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[int(0.975 * iters)]
    return point, lo, hi


def mcnemar(a: list[int], b: list[int]) -> tuple[int, int, float]:
    """McNemar exact-ish test on paired binary outcomes.

    Returns (b01, b10, two-sided p) where b01 = a-fail&b-success,
    b10 = a-success&b-fail.  Uses the binomial test on discordant pairs.
    """
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    # two-sided exact binomial p at q=0.5
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i) * 0.5 ** n
    return b01, b10, min(1.0, 2 * p)
