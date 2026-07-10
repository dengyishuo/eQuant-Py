"""Alpha factor primitives — eAlpha101 equivalent.

These 18 low-level functions are the building blocks for all 101
WorldQuant alpha formulas. They operate on 1-D numpy arrays (typically
within a groupby-apply pattern).

Primitives
----------
cs_rank       : Cross-sectional percentile rank [0, 1]
ts_rank       : Time-series percentile rank within rolling window
ts_max        : Rolling maximum
ts_min        : Rolling minimum
ts_argmax     : Position (lag) of rolling maximum
ts_argmin     : Position (lag) of rolling minimum
ts_stddev     : Rolling standard deviation
ts_sum        : Rolling sum
ts_product    : Rolling product
decay_linear  : Linearly decaying weighted moving average
delay         : Lagged values
delta         : Difference over d periods
signedpower   : sign(x) * |x|^a
scale_alpha   : Scale to unit sum of absolute values
correlation   : Rolling Pearson correlation
covariance    : Rolling covariance
adv           : Average daily volume (rolling mean of volume)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════
# cs_rank — Cross-sectional percentile rank
# ══════════════════════════════════════════════════════════════════════════


def cs_rank(x: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank in [0, 1].

    NAs are preserved. Uses average tie-breaking.
    """
    mask = np.isnan(x)
    result = np.full(len(x), np.nan)
    valid_idx = np.where(~mask)[0]
    if len(valid_idx) < 1:
        return result
    ranks = pd.Series(x[valid_idx]).rank(pct=True, method="average").values
    result[valid_idx] = ranks
    return result


# ══════════════════════════════════════════════════════════════════════════
# ts_rank — Rolling percentile rank
# ══════════════════════════════════════════════════════════════════════════


def ts_rank(x: np.ndarray, d: int) -> np.ndarray:
    """Time-series percentile rank of the current value within the trailing *d* window.

    Returns values in [0, 1]. Leading *d-1* values are NaN.
    """
    d = int(np.floor(d))
    if d <= 1 or len(x) < d:
        return np.full(len(x), np.nan)

    result = np.full(len(x), np.nan)

    def _rank_last(w):
        if np.all(np.isnan(w)):
            return np.nan
        r = pd.Series(w).rank(pct=True, method="average", na_option="bottom").values
        return r[-1]

    rolled = pd.Series(x).rolling(window=d, min_periods=d)
    result = rolled.apply(_rank_last, raw=True).values
    return result


# ══════════════════════════════════════════════════════════════════════════
# ts_max / ts_min — Rolling max / min
# ══════════════════════════════════════════════════════════════════════════


def ts_max(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling maximum over trailing *d* periods."""
    d = int(np.floor(d))
    if d <= 1 or len(x) < d:
        return np.full(len(x), np.nan)
    return pd.Series(x).rolling(window=d, min_periods=d).max().values


def ts_min(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling minimum over trailing *d* periods."""
    d = int(np.floor(d))
    if d <= 1 or len(x) < d:
        return np.full(len(x), np.nan)
    return pd.Series(x).rolling(window=d, min_periods=d).min().values


# ══════════════════════════════════════════════════════════════════════════
# ts_argmax / ts_argmin — Position of rolling extreme
# ══════════════════════════════════════════════════════════════════════════


def ts_argmax(x: np.ndarray, d: int) -> np.ndarray:
    """Days-since-max within trailing *d* window.

    0 = today is the max, 1 = yesterday was the max, etc.
    Ties report the earliest (most distant) occurrence.
    """
    d = int(np.floor(d))
    if d <= 1 or len(x) < d:
        return np.full(len(x), np.nan)

    result = np.full(len(x), np.nan)
    for i in range(d - 1, len(x)):
        w = x[i - d + 1 : i + 1]
        if np.all(np.isnan(w)):
            continue
        # which.max from right → earliest max = smallest index
        valid = ~np.isnan(w)
        valid_vals = w[valid]
        max_val = np.max(valid_vals)
        max_pos = np.where(valid & (w == max_val))[0][0]  # earliest
        result[i] = d - 1 - max_pos  # days ago
    return result


def ts_argmin(x: np.ndarray, d: int) -> np.ndarray:
    """Days-since-min within trailing *d* window."""
    d = int(np.floor(d))
    if d <= 1 or len(x) < d:
        return np.full(len(x), np.nan)

    result = np.full(len(x), np.nan)
    for i in range(d - 1, len(x)):
        w = x[i - d + 1 : i + 1]
        if np.all(np.isnan(w)):
            continue
        valid = ~np.isnan(w)
        valid_vals = w[valid]
        min_val = np.min(valid_vals)
        min_pos = np.where(valid & (w == min_val))[0][0]
        result[i] = d - 1 - min_pos
    return result


# ══════════════════════════════════════════════════════════════════════════
# ts_stddev — Rolling standard deviation
# ══════════════════════════════════════════════════════════════════════════


def ts_stddev(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling standard deviation over trailing *d* periods (sample std)."""
    d = int(np.floor(d))
    if d <= 1 or len(x) < d:
        return np.full(len(x), np.nan)
    return pd.Series(x).rolling(window=d, min_periods=d).std(ddof=1).values


# ══════════════════════════════════════════════════════════════════════════
# ts_sum — Rolling sum
# ══════════════════════════════════════════════════════════════════════════


def ts_sum(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling sum over trailing *d* periods."""
    d = int(np.floor(d))
    if d < 1 or len(x) < d:
        return np.full(len(x), np.nan)
    return pd.Series(x).rolling(window=d, min_periods=d).sum().values


# ══════════════════════════════════════════════════════════════════════════
# ts_product — Rolling product
# ══════════════════════════════════════════════════════════════════════════


def ts_product(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling product over trailing *d* periods."""
    d = int(np.floor(d))
    if d <= 1 or len(x) < d:
        return np.full(len(x), np.nan)

    result = np.full(len(x), np.nan)

    def _prod(w):
        w = w[~np.isnan(w)]
        if len(w) == 0:
            return np.nan
        return np.prod(w)

    rolled = pd.Series(x).rolling(window=d, min_periods=d)
    result = rolled.apply(_prod, raw=True).values
    return result


# ══════════════════════════════════════════════════════════════════════════
# decay_linear — Linearly decaying weighted MA
# ══════════════════════════════════════════════════════════════════════════


def decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    """Weighted moving average with linearly decaying weights.

    Weights: ``d, d-1, ..., 1``, normalized to sum 1.
    Most recent observation gets weight ``d``.
    """
    d = int(np.floor(d))
    if d <= 1:
        return x.copy()
    if len(x) < d:
        return np.full(len(x), np.nan)

    w = np.arange(1, d + 1, dtype=np.float64)  # newest gets weight d
    w /= w.sum()

    result = np.full(len(x), np.nan)
    for i in range(d - 1, len(x)):
        win = x[i - d + 1 : i + 1]
        if np.all(np.isnan(win)):
            continue
        mask = ~np.isnan(win)
        result[i] = np.sum(win[mask] * w[mask]) / np.sum(w[mask])
    return result


# ══════════════════════════════════════════════════════════════════════════
# delay — Lagged values
# ══════════════════════════════════════════════════════════════════════════


def delay(x: np.ndarray, d: int = 1) -> np.ndarray:
    """Lag *x* by *d* periods. Leading *d* values become NaN."""
    d = int(d)
    if d <= 0:
        return x.copy()
    result = np.roll(x, d)
    result[:d] = np.nan
    return result


# ══════════════════════════════════════════════════════════════════════════
# delta — Difference over d periods
# ══════════════════════════════════════════════════════════════════════════


def delta(x: np.ndarray, d: int = 1) -> np.ndarray:
    """Difference: ``x[t] - x[t-d]``."""
    return x - delay(x, d)


# ══════════════════════════════════════════════════════════════════════════
# signedpower — sign(x) * |x|^a
# ══════════════════════════════════════════════════════════════════════════


def signedpower(x: np.ndarray, a: float) -> np.ndarray:
    """Signed power: ``sign(x) * |x|^a``."""
    return np.sign(x) * np.power(np.abs(x), a)


# ══════════════════════════════════════════════════════════════════════════
# scale_alpha — Scale to unit sum of absolute values
# ══════════════════════════════════════════════════════════════════════════


def scale_alpha(x: np.ndarray) -> np.ndarray:
    """Rescale so that sum of absolute values = 1.

    ``x / sum(|x|)``, cross-sectionally or pointwise.
    """
    s = np.nansum(np.abs(x))
    if s < 1e-15:
        return np.full(len(x), np.nan)
    return x / s


# ══════════════════════════════════════════════════════════════════════════
# correlation — Rolling Pearson correlation
# ══════════════════════════════════════════════════════════════════════════


def correlation(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """Rolling Pearson correlation between *x* and *y* over *d* periods."""
    d = int(np.floor(d))
    if d < 3 or len(x) < d:
        return np.full(len(x), np.nan)

    result = np.full(len(x), np.nan)
    for i in range(d - 1, len(x)):
        wx = x[i - d + 1 : i + 1]
        wy = y[i - d + 1 : i + 1]
        mask = ~(np.isnan(wx) | np.isnan(wy))
        if mask.sum() < 3:
            continue
        corr = np.corrcoef(wx[mask], wy[mask])[0, 1]
        result[i] = corr
    return result


# ══════════════════════════════════════════════════════════════════════════
# covariance — Rolling covariance
# ══════════════════════════════════════════════════════════════════════════


def covariance(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """Rolling covariance between *x* and *y* over *d* periods."""
    d = int(np.floor(d))
    if d < 2 or len(x) < d:
        return np.full(len(x), np.nan)

    result = np.full(len(x), np.nan)
    for i in range(d - 1, len(x)):
        wx = x[i - d + 1 : i + 1]
        wy = y[i - d + 1 : i + 1]
        mask = ~(np.isnan(wx) | np.isnan(wy))
        if mask.sum() < 2:
            continue
        result[i] = np.cov(wx[mask], wy[mask], ddof=1)[0, 1]
    return result


# ══════════════════════════════════════════════════════════════════════════
# adv — Average Daily Volume (rolling mean of volume)
# ══════════════════════════════════════════════════════════════════════════


def adv(x: np.ndarray, d: int = 20) -> np.ndarray:
    """Average daily trading volume over *d* periods."""
    d = int(np.floor(d))
    if d < 1 or len(x) < d:
        return np.full(len(x), np.nan)
    return pd.Series(x).rolling(window=d, min_periods=d).mean().values
