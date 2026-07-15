"""Momentum oscillators.

Each function takes a long-format panel DataFrame and returns it
with new indicator columns appended.

Equivalent to eTTR/R/add_RSI.R, add_CCI.R, add_CMO.R, add_TSI.R, etc.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from ettr._panel import _resolve_col
from ettr._rolling import roll_ema, wilder_sum
from equant.utils.panel import slim_output, validate_panel


# ── Helper: typical price ──────────────────────────────────────────────────


def _typical_price(high, low, close):
    return (high + low + close) / 3.0


# ══════════════════════════════════════════════════════════════════════════
# RSI — Relative Strength Index
# ══════════════════════════════════════════════════════════════════════════


def rsi(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: Union[int, Sequence[int]] = 14,
    wilder: bool = True,
    new_col: str = "RSI",
    append: bool = True,
) -> pd.DataFrame:
    """Relative Strength Index.

    Parameters
    ----------
    n : int or sequence
        Lookback period(s).
    wilder : bool
        If True (default), uses Wilder's smoothing for averaging gains/losses.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    ns = [n] if isinstance(n, int) else list(n)
    result = df.copy()

    for period in ns:
        cname = f"{new_col}_{period}"
        result[cname] = np.nan

        for _code, idx in result.groupby("code", sort=False).groups.items():
            vals = result.loc[idx, col].values.astype(np.float64)
            delta = np.diff(vals, prepend=np.nan)
            gain = np.where(delta > 0, delta, 0.0)
            loss = np.where(delta < 0, -delta, 0.0)

            if wilder:
                avg_gain = wilder_sum(gain, period)
                avg_loss = wilder_sum(loss, period)
            else:
                avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
                avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values

            rs = avg_gain / (avg_loss + 1e-15)
            result.loc[idx, cname] = 100.0 - 100.0 / (1.0 + rs)

    return slim_output(result, [f"{new_col}_{p}" for p in ns], append)


# ══════════════════════════════════════════════════════════════════════════
# CCI — Commodity Channel Index
# ══════════════════════════════════════════════════════════════════════════


def cci(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    n: int = 20,
    c: float = 0.015,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Commodity Channel Index.

    ``CCI = (TP - SMA(TP)) / (c * mean_absolute_deviation(TP))``
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    cname = new_col or f"CCI_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        tp = _typical_price(sub[hcol].values, sub[lcol].values, sub[ccol].values)
        tp_sma = pd.Series(tp).rolling(window=n, min_periods=n).mean().values
        mad = pd.Series(np.abs(tp - tp_sma)).rolling(window=n, min_periods=n).mean().values
        result.loc[idx, cname] = np.where(mad > 0, (tp - tp_sma) / (c * mad), 0.0)

    return slim_output(result, cname, append)


# ══════════════════════════════════════════════════════════════════════════
# CMO — Chande Momentum Oscillator
# ══════════════════════════════════════════════════════════════════════════


def cmo(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 14,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Chande Momentum Oscillator.

    ``CMO = 100 * (sum_up - sum_down) / (sum_up + sum_down)``
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or f"CMO_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        delta = np.diff(vals, prepend=np.nan)
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        sum_up = pd.Series(up).rolling(window=n, min_periods=n).sum().values
        sum_down = pd.Series(down).rolling(window=n, min_periods=n).sum().values
        result.loc[idx, cname] = 100.0 * (sum_up - sum_down) / (sum_up + sum_down + 1e-15)

    return slim_output(result, cname, append)


# ══════════════════════════════════════════════════════════════════════════
# TSI — True Strength Index
# ══════════════════════════════════════════════════════════════════════════


def tsi(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n_fast: int = 13,
    n_slow: int = 25,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """True Strength Index.

    ``TSI = 100 * EMA(EMA(momentum, fast), slow) / EMA(EMA(|momentum|, fast), slow)``
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or "TSI"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        mom = np.full(len(vals), np.nan)
        mom[1:] = vals[1:] - vals[:-1]

        e1 = roll_ema(mom, n_fast)
        e2 = roll_ema(e1, n_slow)

        abs_mom = np.abs(mom)
        ae1 = roll_ema(abs_mom, n_fast)
        ae2 = roll_ema(ae1, n_slow)

        result.loc[idx, cname] = 100.0 * e2 / (ae2 + 1e-15)

    return slim_output(result, cname, append)


# ══════════════════════════════════════════════════════════════════════════
# SMI — Stochastic Momentum Index
# ══════════════════════════════════════════════════════════════════════════


def smi(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    n: int = 13,
    n_fast: int = 5,
    n_signal: int = 8,
    wilder: bool = False,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Stochastic Momentum Index.

    Produces ``SMI`` and ``SMI_signal``.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    out_cols = ["SMI", "SMI_signal"]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        high = sub[hcol].values
        low = sub[lcol].values
        cl = sub[ccol].values

        # HH and LL over n
        hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
        ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
        mid = (hh + ll) / 2.0
        dist = cl - mid

        # Double-smoothed
        d1 = roll_ema(dist, n_fast, wilder)
        d2 = roll_ema(d1, n_signal, wilder)

        hh_ema = roll_ema(hh - ll, n_fast, wilder)
        hh_ema2 = roll_ema(hh_ema, n_signal, wilder)

        smi_vals = 200.0 * d2 / (hh_ema2 + 1e-15)
        sig = roll_ema(smi_vals, n_signal, wilder)

        result.loc[idx, "SMI"] = smi_vals
        result.loc[idx, "SMI_signal"] = sig

    return slim_output(result, out_cols, append)


# ══════════════════════════════════════════════════════════════════════════
# WPR — Williams %R
# ══════════════════════════════════════════════════════════════════════════


def wpr(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    n: int = 14,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Williams %R.

    ``%R = -100 * (HH - Close) / (HH - LL)``
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    cname = new_col or f"WPR_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        hh = sub[hcol].rolling(window=n, min_periods=n).max().values
        ll = sub[lcol].rolling(window=n, min_periods=n).min().values
        cl = sub[ccol].values
        spread = hh - ll
        result.loc[idx, cname] = np.where(spread > 0, -100.0 * (hh - cl) / spread, 0.0)

    return slim_output(result, cname, append)


# ══════════════════════════════════════════════════════════════════════════
# Ultimate Oscillator
# ══════════════════════════════════════════════════════════════════════════


def ultimate_oscillator(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    n1: int = 7,
    n2: int = 14,
    n3: int = 28,
    w1: float = 4.0,
    w2: float = 2.0,
    w3: float = 1.0,
    new_col: str = "Ultimate_Osc",
    append: bool = True,
) -> pd.DataFrame:
    """Ultimate Oscillator.

    Uses three timeframes weighted by w1, w2, w3.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    result = df.copy()
    result[new_col] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        high = sub[hcol].values
        low = sub[lcol].values
        cl = sub[ccol].values

        # True low and buying pressure
        tlow = np.minimum(low[1:], cl[:-1])
        bp = cl[1:] - tlow
        tr = np.maximum(high[1:], cl[:-1]) - np.minimum(low[1:], cl[:-1])

        avg1 = pd.Series(bp).rolling(n1, min_periods=n1).sum().values / pd.Series(tr).rolling(n1, min_periods=n1).sum().values
        avg2 = pd.Series(bp).rolling(n2, min_periods=n2).sum().values / pd.Series(tr).rolling(n2, min_periods=n2).sum().values
        avg3 = pd.Series(bp).rolling(n3, min_periods=n3).sum().values / pd.Series(tr).rolling(n3, min_periods=n3).sum().values

        uo = 100.0 * (w1 * avg1 + w2 * avg2 + w3 * avg3) / (w1 + w2 + w3)
        result.loc[idx, new_col] = np.concatenate([[np.nan], uo])

    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════
# ROC / Momentum
# ══════════════════════════════════════════════════════════════════════════


def roc(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 10,
    type: str = "continuous",
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Rate of Change.

    Parameters
    ----------
    type : str
        ``"continuous"`` = log return, ``"discrete"`` = simple return.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or f"ROC_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        if type == "continuous":
            result.loc[idx, cname] = pd.Series(vals).pct_change(periods=n).values
        else:
            shifted = np.roll(vals, n)
            shifted[:n] = np.nan
            result.loc[idx, cname] = (vals - shifted) / np.abs(shifted + 1e-15)

    return slim_output(result, cname, append)


def momentum(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 10,
    na_pad: bool = True,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Price momentum: ``close - close[n periods ago]``."""
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or f"MOM_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        mom = vals - np.roll(vals, n)
        mom[:n] = np.nan if na_pad else mom[:n]
        result.loc[idx, cname] = mom

    return slim_output(result, cname, append)


# ══════════════════════════════════════════════════════════════════════════
# CTI — Correlation Trend Indicator
# ══════════════════════════════════════════════════════════════════════════


def cti(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 10,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Correlation Trend Indicator.

    Pearson correlation of price with ``1, 2, ..., n`` over the trailing window.
    Close to +1 = uptrend, close to -1 = downtrend.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or f"CTI_{n}"
    result = df.copy()
    result[cname] = np.nan

    x = np.arange(1, n + 1, dtype=np.float64)
    x_mean = x.mean()
    x_std = x.std(ddof=0)

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        rolled = pd.Series(vals).rolling(window=n, min_periods=n)

        def _corr(w):
            w = w.values
            y_mean = w.mean()
            y_std = w.std(ddof=0)
            if y_std < 1e-15:
                return 0.0
            return ((w - y_mean) * (x - x_mean)).mean() / (x_std * y_std)

        result.loc[idx, cname] = rolled.apply(_corr, raw=False).values

    return slim_output(result, cname, append)


# ══════════════════════════════════════════════════════════════════════════
# RVI — Relative Vigor Index
# ══════════════════════════════════════════════════════════════════════════


def rvi(
    df: pd.DataFrame,
    open_col: Optional[str] = None,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    n: int = 10,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Relative Vigor Index.

    Produces ``RVI`` and ``RVI_signal``.
    """
    validate_panel(df)
    ocol = _resolve_col(df, "open", open_col)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    out_cols = [f"RVI_{n}", f"RVI_{n}_signal"]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        o = sub[ocol].values
        h = sub[hcol].values
        l = sub[lcol].values
        c = sub[ccol].values

        numerator = c - o
        denominator = h - l
        num_4 = pd.Series(numerator).rolling(4, min_periods=4).mean().values
        den_4 = pd.Series(denominator).rolling(4, min_periods=4).mean().values

        rvi_raw = num_4 / (den_4 + 1e-15)
        rvi_vals = pd.Series(rvi_raw).rolling(n, min_periods=n).mean().values
        sig = (rvi_vals + 2 * np.roll(rvi_vals, 1) + 2 * np.roll(rvi_vals, 2) + np.roll(rvi_vals, 3)) / 6.0

        result.loc[idx, out_cols[0]] = rvi_vals
        result.loc[idx, out_cols[1]] = sig

    return slim_output(result, out_cols, append)


# ══════════════════════════════════════════════════════════════════════════
# DVI — Dynamic Volatility Index
# ══════════════════════════════════════════════════════════════════════════


def dvi(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 252,
    n_smooth: int = 5,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Dynamic Volatility Index.

    Compares recent volatility to long-term volatility.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or f"DVI_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        rets = np.full(len(vals), np.nan)
        rets[1:] = vals[1:] - vals[:-1]  # Simple diff approximation

        sd_short = pd.Series(rets).rolling(window=n_smooth, min_periods=n_smooth).std().values
        sd_long = pd.Series(rets).rolling(window=n, min_periods=n).std().values
        result.loc[idx, cname] = sd_short / (sd_long + 1e-15) * 100.0

    return slim_output(result, cname, append)


# ══════════════════════════════════════════════════════════════════════════
# StochRSI / Stoch
# ══════════════════════════════════════════════════════════════════════════


def stoch(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    n_fast_k: int = 14,
    n_fast_d: int = 3,
    n_slow_d: int = 3,
    new_col: str = "Stoch",
    append: bool = True,
) -> pd.DataFrame:
    """Stochastic Oscillator (Fast and Slow).

    Produces ``{new_col}_fastK``, ``{new_col}_fastD``, ``{new_col}_slowD``.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    out_cols = [f"{new_col}_fastK", f"{new_col}_fastD", f"{new_col}_slowD"]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        hh = sub[hcol].rolling(n_fast_k, min_periods=n_fast_k).max().values
        ll = sub[lcol].rolling(n_fast_k, min_periods=n_fast_k).min().values
        cl = sub[ccol].values

        spread = hh - ll
        fast_k = np.where(spread > 0, 100.0 * (cl - ll) / spread, 50.0)
        fast_d = pd.Series(fast_k).rolling(n_fast_d, min_periods=n_fast_d).mean().values
        slow_d = pd.Series(fast_d).rolling(n_slow_d, min_periods=n_slow_d).mean().values

        result.loc[idx, out_cols[0]] = fast_k
        result.loc[idx, out_cols[1]] = fast_d
        result.loc[idx, out_cols[2]] = slow_d

    return slim_output(result, out_cols, append)


# ══════════════════════════════════════════════════════════════════════════
# KDJ
# ══════════════════════════════════════════════════════════════════════════


def kdj(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    n: int = 9,
    k_smooth: int = 3,
    d_smooth: int = 3,
    new_col: str = "KDJ",
    append: bool = True,
) -> pd.DataFrame:
    """KDJ indicator (Chinese market variant of Stochastic).

    Produces ``{new_col}_K``, ``{new_col}_D``, ``{new_col}_J``.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    out_cols = [f"{new_col}_K", f"{new_col}_D", f"{new_col}_J"]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        ll = sub[lcol].rolling(n, min_periods=n).min().values
        hh = sub[hcol].rolling(n, min_periods=n).max().values
        cl = sub[ccol].values

        spread = hh - ll
        rsv = np.where(spread > 0, 100.0 * (cl - ll) / spread, 50.0)

        # K and D are EMA-like smoothed RSV
        k = roll_ema(rsv, k_smooth)
        d = roll_ema(k, d_smooth)
        j = 3.0 * k - 2.0 * d

        result.loc[idx, out_cols[0]] = k
        result.loc[idx, out_cols[1]] = d
        result.loc[idx, out_cols[2]] = j

    return slim_output(result, out_cols, append)
