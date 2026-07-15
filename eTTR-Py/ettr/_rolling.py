"""Numba-accelerated rolling-window primitives.

These functions replace the C routines in eTTR/src/ and provide
O(n) or O(n log n) implementations of common sliding-window operations
that are either not available in pandas or are slower in pure Python.

All functions operate on 1-D numpy arrays and return same-length arrays
with leading NaN padding (matching the xts convention).
"""

from __future__ import annotations

import numpy as np
import numba as nb

# NaN sentinel matching IEEE 754 double
_NAN = np.nan


# ═══════════════════════════════════════════════════════════════════
# Run Sum  (runfun.c: runsum)
# ═══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def roll_sum(x: np.ndarray, window: int) -> np.ndarray:
    """O(n) sliding sum.

    Equivalent to ``ettr::runSum`` / ``pandas.rolling().sum()`` but
    provides the exact NaN-handling behaviour of the original C code.

    Leading *window* - 1 values are NaN.  Internal NaN values in *x*
    trigger a full recomputation of the window (matching xts behaviour).
    """
    n = len(x)
    out = np.full(n, _NAN)
    if n < window or window < 1:
        return out

    # Find first non-NaN position
    first = 0
    while first < n and np.isnan(x[first]):
        first += 1

    if first + window > n:
        return out  # not enough data

    # Seed sum
    s = 0.0
    for i in range(first, first + window):
        s += x[i]
    out[first + window - 1] = s

    # Sliding window
    for i in range(first + window, n):
        if np.isnan(x[i]):
            # Recompute window sum (matching xts behaviour)
            s = 0.0
            for j in range(i - window + 1, i + 1):
                if not np.isnan(x[j]):
                    s += x[j]
        else:
            s += x[i]
            if not np.isnan(x[i - window]):
                s -= x[i - window]
        out[i] = s

    return out


# ═══════════════════════════════════════════════════════════════════
# Run Min  (runfun.c: runmin)
# ═══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def roll_min(x: np.ndarray, window: int) -> np.ndarray:
    """O(n*window) naive sliding minimum.

    For typical financial window sizes (5-252), the O(n*w) approach
    with tight C loops via numba is fast enough.
    """
    n = len(x)
    out = np.full(n, _NAN)
    if n < window or window < 1:
        return out

    first = 0
    while first < n and np.isnan(x[first]):
        first += 1
    if first + window > n:
        return out

    # Seed
    vmin = x[first]
    for i in range(first, first + window):
        if x[i] < vmin or np.isnan(vmin):
            vmin = x[i]
    out[first + window - 1] = vmin

    # Sliding
    for i in range(first + window, n):
        vmin = x[i]
        for j in range(1, window):
            if x[i - j] < vmin or np.isnan(vmin):
                vmin = x[i - j]
        out[i] = vmin

    return out


# ═══════════════════════════════════════════════════════════════════
# Run Max  (runfun.c: runmax)
# ═══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def roll_max(x: np.ndarray, window: int) -> np.ndarray:
    """O(n*window) naive sliding maximum."""
    n = len(x)
    out = np.full(n, _NAN)
    if n < window or window < 1:
        return out

    first = 0
    while first < n and np.isnan(x[first]):
        first += 1
    if first + window > n:
        return out

    vmax = x[first]
    for i in range(first, first + window):
        if x[i] > vmax or np.isnan(vmax):
            vmax = x[i]
    out[first + window - 1] = vmax

    for i in range(first + window, n):
        vmax = x[i]
        for j in range(1, window):
            if x[i - j] > vmax or np.isnan(vmax):
                vmax = x[i - j]
        out[i] = vmax

    return out


# ═══════════════════════════════════════════════════════════════════
# Run Median  (runfun.c: runmedian)
# ═══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def _median_of(arr: np.ndarray, n_valid: int) -> float:
    """Compute median of first *n_valid* elements of *arr* in-place (sorts)."""
    sub = arr[:n_valid].copy()
    sub.sort()
    if n_valid % 2 == 1:
        return sub[n_valid // 2]
    else:
        return (sub[n_valid // 2 - 1] + sub[n_valid // 2]) / 2.0


@nb.njit(cache=True)
def roll_median(x: np.ndarray, window: int) -> np.ndarray:
    """Sliding median over *window* observations.

    Uses full sort per window (O(n * w log w)). For typical window sizes
    this is acceptable with numba JIT.
    """
    n = len(x)
    out = np.full(n, _NAN)
    if n < window or window < 1:
        return out

    first = 0
    while first < n and np.isnan(x[first]):
        first += 1
    if first + window > n:
        return out

    buf = np.empty(window, dtype=x.dtype)

    for i in range(first + window - 1, n):
        k = 0
        for j in range(i - window + 1, i + 1):
            if not np.isnan(x[j]):
                buf[k] = x[j]
                k += 1
        if k >= 1:
            out[i] = _median_of(buf, k)

    return out


# ═══════════════════════════════════════════════════════════════════
# Run MAD (Median Absolute Deviation)  (runfun.c: runmad)
# ═══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def _median_sorted(arr: np.ndarray, n: int) -> float:
    sub = np.sort(arr[:n])
    if n % 2 == 1:
        return sub[n // 2]
    else:
        return (sub[n // 2 - 1] + sub[n // 2]) / 2.0


@nb.njit(cache=True)
def roll_mad(x: np.ndarray, window: int, center: np.ndarray | None = None) -> np.ndarray:
    """Rolling median absolute deviation.

    Parameters
    ----------
    x : array
        Input data.
    window : int
        Lookback window.
    center : array or None
        Central tendency per observation. If None, uses the rolling median.
    """
    n = len(x)
    out = np.full(n, _NAN)
    if n < window or window < 1:
        return out

    first = 0
    while first < n and np.isnan(x[first]):
        first += 1

    abs_dev = np.empty(window, dtype=x.dtype)

    for i in range(first + window - 1, n):
        ctr = center[i] if center is not None else _NAN
        if center is None:
            # Use rolling median as center — two-pass approach
            buf = np.empty(window, dtype=x.dtype)
            k = 0
            for j in range(i - window + 1, i + 1):
                if not np.isnan(x[j]):
                    buf[k] = x[j]
                    k += 1
            if k < 1:
                continue
            ctr = _median_sorted(buf, k)

        if np.isnan(ctr):
            continue

        k = 0
        for j in range(i - window + 1, i + 1):
            if not np.isnan(x[j]):
                abs_dev[k] = abs(x[j] - ctr)
                k += 1
        if k >= 1:
            out[i] = _median_sorted(abs_dev, k)

    return out


# ═══════════════════════════════════════════════════════════════════
# Wilder Sum  (WilderSum.c)
# ═══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def wilder_sum(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder's smoothed sum.

    ``result[i] = x[i] + result[i-1] * (n-1)/n``

    Used as the foundation for ATR, RSI (Wilder variant), ADX, etc.
    """
    nr = len(x)
    out = np.full(nr, _NAN)

    # Find first non-NA
    beg = n - 1
    for i in range(beg):
        if np.isnan(x[i]):
            out[i] = _NAN
            beg += 1
        else:
            out[i] = _NAN
    if beg >= nr:
        return out

    # Seed: sum of first n values
    s = 0.0
    for i in range(beg - n + 1, beg + 1):
        s += x[i]
    out[beg] = (x[beg] + s * (n - 1) / n)

    # Recurrence
    for i in range(beg + 1, nr):
        out[i] = x[i] + out[i - 1] * (n - 1) / n

    return out


# ═══════════════════════════════════════════════════════════════════
# EMA (Exponential Moving Average)  (moving_averages.c: ema)
# ═══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def roll_ema(x: np.ndarray, n: int, wilder: bool = False) -> np.ndarray:
    """Exponential moving average.

    Parameters
    ----------
    x : array
        Input data.
    n : int
        Lookback period.
    wilder : bool
        If True, uses Wilder's ratio ``1/n`` instead of ``2/(n+1)``.
    """
    nr = len(x)
    out = np.full(nr, _NAN)
    ratio = 1.0 / n if wilder else 2.0 / (n + 1)

    # Find first non-NA
    first = 0
    while first < nr and np.isnan(x[first]):
        first += 1
    if first + n > nr:
        return out

    # Seed: simple mean of first n values
    seed = 0.0
    for i in range(first, first + n):
        seed += x[i] / n
    out[first + n - 1] = seed

    # EMA recurrence
    for i in range(first + n, nr):
        out[i] = x[i] * ratio + out[i - 1] * (1.0 - ratio)

    return out


# ═══════════════════════════════════════════════════════════════════
# WMA (Weighted Moving Average)  (moving_averages.c: wma)
# ═══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def roll_wma(x: np.ndarray, n: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted moving average.

    If *weights* is None, uses linearly increasing weights ``1, 2, ..., n``
    (the default in eTTR).
    """
    nr = len(x)
    out = np.full(nr, _NAN)

    if weights is None:
        wts = np.arange(1, n + 1, dtype=np.float64)
    else:
        wts = weights[:n].astype(np.float64)

    wtsum = np.sum(wts)
    if wtsum == 0.0:
        return out

    first = 0
    while first < nr and np.isnan(x[first]):
        first += 1
    if first + n > nr:
        return out

    begin = first + n - 1
    for i in range(begin, nr):
        num = 0.0
        ni = i - n + 1
        for j in range(n):
            num += x[ni + j] * wts[j]
        out[i] = num / wtsum

    return out


# ═══════════════════════════════════════════════════════════════════
# ZLEMA (Zero-Lag EMA)  (moving_averages.c: zlema)
# ═══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def roll_zlema(x: np.ndarray, n: int) -> np.ndarray:
    """Zero-lag exponential moving average."""
    nr = len(x)
    out = np.full(nr, _NAN)
    ratio = 2.0 / (n + 1)
    lag = int(1.0 / ratio)
    wt = 1.0 / ratio - float(lag)
    w1 = 1.0 - wt

    first = 0
    while first < nr and np.isnan(x[first]):
        first += 1
    if first + n > nr or first + lag + 1 >= nr:
        return out

    # Seed EMA
    seed = 0.0
    for i in range(first, first + n):
        seed += x[i] / n
    out[first + n - 1] = seed

    for i in range(first + n, nr):
        loc = i - lag
        if loc < 0 or loc + 1 >= nr:
            continue
        value = 2.0 * x[i] - (w1 * x[loc] + wt * x[loc + 1])
        out[i] = ratio * value + (1.0 - ratio) * out[i - 1]

    return out


# ═══════════════════════════════════════════════════════════════════
# EVWMA (Elastic Volume-Weighted MA)  (moving_averages.c: evwma)
# ═══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def roll_evwma(price: np.ndarray, volume: np.ndarray, n: int) -> np.ndarray:
    """Elastic volume-weighted moving average."""
    nr = len(price)
    out = np.full(nr, _NAN)

    first_p = 0
    while first_p < nr and np.isnan(price[first_p]):
        first_p += 1
    first_v = 0
    while first_v < nr and np.isnan(volume[first_v]):
        first_v += 1
    first = max(first_p, first_v)

    begin = first + n - 1
    if begin >= nr:
        return out

    out[begin] = price[begin]

    # Initial volume sum
    vol_sum = 0.0
    for i in range(first, begin + 1):
        vol_sum += volume[i]

    for i in range(begin + 1, nr):
        vol_sum = vol_sum + volume[i] - volume[i - n]
        out[i] = ((vol_sum - volume[i]) * out[i - 1] + volume[i] * price[i]) / vol_sum

    return out


# ═══════════════════════════════════════════════════════════════════
# Run Percent Rank  (percent_rank.c: ettr_rollPercentRank)
# ═══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def roll_percent_rank(x: np.ndarray, window: int, cumulative: bool = False) -> np.ndarray:
    """Rolling percentile rank.

    For each position *i*, computes the fraction of values in the trailing
    *window* that are strictly less than ``x[i]``.

    Parameters
    ----------
    x : array
        Input data.
    window : int
        Lookback window (fixed) or ignored if *cumulative* is True.
    cumulative : bool
        If True, uses an expanding window from the first observation.
    """
    n = len(x)
    out = np.full(n, _NAN)

    # Find first position after leading NAs
    beg = window - 1
    n_na = 0
    for i in range(beg):
        if np.isnan(x[i]):
            beg += 1
            n_na += 1
            if beg >= n:
                return out

    if cumulative:
        out[beg] = 0.5  # single value: rank at 0.5
        for i in range(beg + 1, n):
            if np.isnan(x[i]):
                continue
            n_less = 0.0
            for j in range(i):
                if not np.isnan(x[j]):
                    diff = x[j] - x[i]
                    if diff < 0:
                        n_less += 1.0
                    elif abs(diff) < 1e-8:
                        n_less += 0.5
            out[i] = n_less / (i + 1)
    else:
        for i in range(beg, n):
            if np.isnan(x[i]):
                continue
            n_less = 0.0
            j1 = i - window + 1
            for j in range(j1, i):
                if np.isnan(x[j]):
                    continue
                diff = x[j] - x[i]
                if diff < 0:
                    n_less += 1.0
                elif abs(diff) < 1e-8:
                    n_less += 0.5
            out[i] = n_less / window

    return out


# ═══════════════════════════════════════════════════════════════════
# Rolling Covariance  (runfun.c: runcov)
# ═══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def roll_cov(x: np.ndarray, y: np.ndarray, window: int, sample: bool = True) -> np.ndarray:
    """Rolling covariance between *x* and *y*.

    Leading ``first + window - 1`` values are NaN.
    """
    nr = len(x)
    out = np.full(nr, _NAN)
    if nr != len(y) or nr < window:
        return out

    first_x = 0
    while first_x < nr and np.isnan(x[first_x]):
        first_x += 1
    first_y = 0
    while first_y < nr and np.isnan(y[first_y]):
        first_y += 1
    first = max(first_x, first_y)

    first_i = first + window - 1
    if first_i >= nr:
        return out

    denom = (window - 1) if sample else window
    if denom <= 0:
        return out

    buf_x = np.empty(window, dtype=np.float64)
    buf_y = np.empty(window, dtype=np.float64)

    for i in range(first_i, nr):
        k = 0
        for j in range(i - window + 1, i + 1):
            if not (np.isnan(x[j]) or np.isnan(y[j])):
                buf_x[k] = x[j]
                buf_y[k] = y[j]
                k += 1
        if k < 2:
            continue

        mu_x = 0.0
        mu_y = 0.0
        for j in range(k):
            mu_x += buf_x[j] / k
            mu_y += buf_y[j] / k

        cov_val = 0.0
        for j in range(k):
            cov_val += (buf_x[j] - mu_x) * (buf_y[j] - mu_y)
        out[i] = cov_val / denom

    return out


# ══════════════════════════════════════════════════════════════════
# Run SD / Var — derived from pandas but included for completeness
# ══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def roll_sd(x: np.ndarray, window: int, sample: bool = True) -> np.ndarray:
    """Rolling standard deviation."""
    nr = len(x)
    out = np.full(nr, _NAN)
    if nr < window or window < 2:
        return out

    first = 0
    while first < nr and np.isnan(x[first]):
        first += 1
    if first + window > nr:
        return out

    denom = (window - 1) if sample else window

    for i in range(first + window - 1, nr):
        # Compute mean
        s = 0.0
        k = 0
        for j in range(i - window + 1, i + 1):
            if not np.isnan(x[j]):
                s += x[j]
                k += 1
        if k < 2:
            continue
        mu = s / k

        # Compute variance
        v = 0.0
        for j in range(i - window + 1, i + 1):
            if not np.isnan(x[j]):
                d = x[j] - mu
                v += d * d
        out[i] = np.sqrt(v / denom)

    return out


# ═══════════════════════════════════════════════════════════════════
# Aroon indicators  (aroon.c: aroon_max)
# ═══════════════════════════════════════════════════════════════════


@nb.njit(cache=True)
def aroon_max(x_high: np.ndarray, x_low: np.ndarray, n: int):
    """Compute aroon_up and aroon_down.

    Returns (aroon_up, aroon_down, oscillator) as a tuple of arrays.
    """
    nr = len(x_high)
    aroon_up = np.full(nr, _NAN)
    aroon_down = np.full(nr, _NAN)
    osc = np.full(nr, _NAN)

    for i in range(n, nr):
        # Find index of max high and min low in window
        hmax = x_high[i - n]
        hmax_idx = 0
        lmin = x_low[i - n]
        lmin_idx = 0
        for j in range(1, n + 1):
            idx = i - n + j
            if not np.isnan(x_high[idx]) and x_high[idx] > hmax:
                hmax = x_high[idx]
                hmax_idx = j
            if not np.isnan(x_low[idx]) and x_low[idx] < lmin:
                lmin = x_low[idx]
                lmin_idx = j

        aroon_up[i] = 100.0 * (n - hmax_idx) / n
        aroon_down[i] = 100.0 * (n - lmin_idx) / n
        osc[i] = aroon_up[i] - aroon_down[i]

    return aroon_up, aroon_down, osc


# ═══════════════════════════════════════════════════════════════════
# Helpers for panel (groupby-apply) usage
# ═══════════════════════════════════════════════════════════════════


def _validate_univariate(x: np.ndarray, name: str = "x") -> None:
    """Raise if *x* is not a 1-D float array."""
    if not isinstance(x, np.ndarray) or x.ndim != 1:
        raise TypeError(f"'{name}' must be a 1-D numpy array, got shape {getattr(x, 'shape', ())}")
    if x.dtype.kind not in ("f", "i"):
        raise TypeError(f"'{name}' must be float/int, got {x.dtype}")
