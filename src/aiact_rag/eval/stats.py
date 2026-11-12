"""Bootstrap confidence intervals, so small differences between configurations are not over-read."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

N_BOOT = 5000
SEED = 1689  # the regulation number; fixed so reports are reproducible


def mean_ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    boots = x[rng.integers(0, x.size, size=(N_BOOT, x.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_delta_ci(
    a: Sequence[float], b: Sequence[float], alpha: float = 0.05
) -> tuple[float, float, float, float]:
    """Mean of (b - a) over paired items, its CI, and a two-sided bootstrap p-value."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return float("nan"), float("nan"), float("nan"), float("nan")
    d = np.asarray([y - x for x, y in pairs], dtype=float)
    rng = np.random.default_rng(SEED)
    boots = d[rng.integers(0, d.size, size=(N_BOOT, d.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))
