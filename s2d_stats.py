"""
s2d_stats.py
============
Aggregation statistics for the sparse→dense study.

rliable was attempted (`pip install rliable`) but its `arch` dependency is
incompatible with this environment (deprecate_kwarg TypeError at import), so per
the study spec we implement the IQM + stratified percentile bootstrap manually
(10,000 resamples). Point estimates alone are not reported.
"""

import numpy as np
from scipy import stats as _sps

BOOTSTRAP_REPS = 10_000


def iqm(x: np.ndarray) -> float:
    """Interquartile mean = 25%-trimmed mean (robust central tendency)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return np.nan
    if len(x) < 4:
        return float(np.mean(x))  # trimming ill-defined for very few samples
    return float(_sps.trim_mean(x, 0.25))


def bootstrap_ci(x: np.ndarray, agg=iqm, reps: int = BOOTSTRAP_REPS,
                 alpha: float = 0.05, seed: int = 0):
    """
    Stratified percentile bootstrap over the seed dimension.

    Returns (point_estimate, ci_low, ci_high) where the point estimate is `agg`
    on the observed samples and the CI is the [α/2, 1−α/2] percentile interval of
    `agg` over `reps` resamples (seeds drawn with replacement).
    """
    x = np.asarray(x, dtype=np.float64)
    point = agg(x)
    n = len(x)
    if n < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    boot = np.array([agg(x[i]) for i in idx])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
