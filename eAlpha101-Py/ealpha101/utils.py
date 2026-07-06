"""
Rolling utility functions for long-format panel DataFrames.

All functions expect the caller to have already called
  df.sort_values(["code", "date"]).groupby("code")
or operate row-wise on a single group.  The public helpers below
accept a Series (within a groupby transform) and return a Series.
"""

import numpy as np
import pandas as pd


# ── Cross-sectional ────────────────────────────────────────────────────────

def cs_rank(s: pd.Series) -> pd.Series:
    """Percentile rank cross-sectionally (pct=True → [0,1])."""
    return s.rank(pct=True)


def scale_alpha(s: pd.Series) -> pd.Series:
    """Scale so that sum of absolute values equals 1."""
    total = s.abs().sum()
    return s / total if total != 0 else s


# ── Time-series (operate within a single stock's sorted series) ────────────

def ts_sum(s: pd.Series, d: int) -> pd.Series:
    return s.rolling(d, min_periods=1).sum()


def ts_mean(s: pd.Series, d: int) -> pd.Series:
    return s.rolling(d, min_periods=1).mean()


def ts_stddev(s: pd.Series, d: int) -> pd.Series:
    return s.rolling(d, min_periods=2).std()


def ts_max(s: pd.Series, d: int) -> pd.Series:
    return s.rolling(d, min_periods=1).max()


def ts_min(s: pd.Series, d: int) -> pd.Series:
    return s.rolling(d, min_periods=1).min()


def ts_rank(s: pd.Series, d: int) -> pd.Series:
    """Rolling rank of the current value within the past d observations (pct)."""
    return s.rolling(d, min_periods=1).rank(pct=True)


def ts_argmax(s: pd.Series, d: int) -> pd.Series:
    """Position (1-indexed from oldest) of the maximum in the rolling window."""
    def _argmax(x):
        if x.isna().all():
            return np.nan
        return float(np.argmax(x.values) + 1)
    return s.rolling(d, min_periods=1).apply(_argmax, raw=False)


def ts_argmin(s: pd.Series, d: int) -> pd.Series:
    """Position (1-indexed from oldest) of the minimum in the rolling window."""
    def _argmin(x):
        if x.isna().all():
            return np.nan
        return float(np.argmin(x.values) + 1)
    return s.rolling(d, min_periods=1).apply(_argmin, raw=False)


def ts_product(s: pd.Series, d: int) -> pd.Series:
    def _prod(x):
        return np.prod(x)
    return s.rolling(d, min_periods=1).apply(_prod, raw=True)


def delay(s: pd.Series, d: int) -> pd.Series:
    return s.shift(d)


def delta(s: pd.Series, d: int) -> pd.Series:
    return s - s.shift(d)


def correlation(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=2).corr(y)


def covariance(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=2).cov(y)


def decay_linear(s: pd.Series, d: int) -> pd.Series:
    """Linearly weighted moving average: weight d for most recent, 1 for oldest."""
    weights = np.arange(1, d + 1, dtype=float)
    weights /= weights.sum()
    def _wma(x):
        n = len(x)
        w = weights[-n:]
        w = w / w.sum()
        return float(np.dot(w, x))
    return s.rolling(d, min_periods=1).apply(_wma, raw=True)


def signedpower(s: pd.Series, exp) -> pd.Series:
    """sign(s) * abs(s) ** exp, element-wise."""
    if isinstance(exp, pd.Series):
        return np.sign(s) * (s.abs() ** exp)
    return np.sign(s) * (s.abs() ** exp)


def adv(volume: pd.Series, d: int) -> pd.Series:
    """d-day average volume."""
    return ts_mean(volume, d)
