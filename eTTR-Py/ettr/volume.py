"""Volume-based indicators.

Each function takes a long-format panel DataFrame and returns it
with new indicator columns appended.

Equivalent to eTTR/R/add_OBV.R, add_CMF.R, add_VWAP.R, etc.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ettr._panel import _resolve_col
from ettr._rolling import roll_ema
from equant.utils.panel import slim_output, validate_panel


# ══════════════════════════════════════════════════════════════════════════
# OBV — On-Balance Volume
# ══════════════════════════════════════════════════════════════════════════


def obv(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    new_col: str = "OBV",
    append: bool = True,
) -> pd.DataFrame:
    """On-Balance Volume.

    ``OBV[i] = OBV[i-1] + sign(close[i] - close[i-1]) * volume[i]``
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    vol_col = _resolve_col(df, "volume", volume_col)
    result = df.copy()
    result[new_col] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        close = sub[col].values
        volume = sub[vol_col].values
        obv_vals = np.zeros(len(close), dtype=np.float64)
        obv_vals[0] = 0.0
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv_vals[i] = obv_vals[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv_vals[i] = obv_vals[i - 1] - volume[i]
            else:
                obv_vals[i] = obv_vals[i - 1]
        # Replace first with NaN
        obv_vals[0] = np.nan
        result.loc[sub.index, new_col] = obv_vals

    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════
# CMF — Chaikin Money Flow
# ══════════════════════════════════════════════════════════════════════════


def cmf(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    n: int = 20,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Chaikin Money Flow.

    ``CMF = sum(MFV, n) / sum(volume, n)`` where
    ``MFV = ((close - low) - (high - close)) / (high - low) * volume``
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    vcol = _resolve_col(df, "volume", volume_col)
    cname = new_col or f"CMF_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        h = sub[hcol].values
        l = sub[lcol].values
        c = sub[ccol].values
        v = sub[vcol].values

        hl_range = h - l
        cl_adj = np.where(hl_range > 0, ((c - l) - (h - c)) / hl_range, 0.0)
        mfv = cl_adj * v

        sum_mfv = pd.Series(mfv).rolling(window=n, min_periods=n).sum().values
        sum_vol = pd.Series(v).rolling(window=n, min_periods=n).sum().values
        result.loc[idx, cname] = sum_mfv / (sum_vol + 1e-15)

    return slim_output(result, cname, append)


# ══════════════════════════════════════════════════════════════════════════
# VWAP — Volume-Weighted Average Price
# ══════════════════════════════════════════════════════════════════════════


def vwap(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    new_col: str = "VWAP",
    append: bool = True,
) -> pd.DataFrame:
    """Cumulative Volume-Weighted Average Price (session VWAP).

    Reset per asset — computes cumulative from the first observation.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    vcol = _resolve_col(df, "volume", volume_col)
    result = df.copy()
    result[new_col] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        tp = (sub[hcol].values + sub[lcol].values + sub[ccol].values) / 3.0
        vol = sub[vcol].values
        cum_pv = np.cumsum(tp * vol)
        cum_v = np.cumsum(vol)
        result.loc[sub.index, new_col] = cum_pv / (cum_v + 1e-15)

    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════
# MFI — Money Flow Index
# ══════════════════════════════════════════════════════════════════════════


def mfi(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    n: int = 14,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Money Flow Index.

    Volume-weighted RSI using typical price.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    vcol = _resolve_col(df, "volume", volume_col)
    cname = new_col or f"MFI_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        tp = (sub[hcol].values + sub[lcol].values + sub[ccol].values) / 3.0
        vol = sub[vcol].values
        rmf = tp * vol

        pos_flow = np.where(tp[1:] > tp[:-1], rmf[1:], 0.0)
        neg_flow = np.where(tp[1:] < tp[:-1], rmf[1:], 0.0)

        sum_pos = pd.Series(np.concatenate([[np.nan], pos_flow])).rolling(n, min_periods=n).sum().values
        sum_neg = pd.Series(np.concatenate([[np.nan], neg_flow])).rolling(n, min_periods=n).sum().values

        mr = sum_pos / (sum_neg + 1e-15)
        result.loc[idx, cname] = 100.0 - 100.0 / (1.0 + mr)

    return slim_output(result, cname, append)


# ══════════════════════════════════════════════════════════════════════════
# EMV — Ease of Movement
# ══════════════════════════════════════════════════════════════════════════


def emv(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    n: int = 14,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Ease of Movement.

    ``EMV = SMA(box_ratio, n)`` where
    ``box_ratio = (midpoint_move / box_ratio_denom)``
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    vcol = _resolve_col(df, "volume", volume_col)
    cname = new_col or f"EMV_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        h = sub[hcol].values
        l = sub[lcol].values
        v = sub[vcol].values

        mid_move = (h[1:] + l[1:]) / 2.0 - (h[:-1] + l[:-1]) / 2.0
        hl_range = h[1:] - l[1:]
        denom = np.where(hl_range > 0, v[1:] / hl_range, np.nan)
        br = np.where(np.abs(denom) > 1e-15, mid_move / denom, 0.0)

        emv_vals = pd.Series(np.concatenate([[np.nan], br])).rolling(n, min_periods=n).mean().values
        result.loc[idx, cname] = emv_vals

    return slim_output(result, cname, append)


# ══════════════════════════════════════════════════════════════════════════
# CLV — Close Location Value
# ══════════════════════════════════════════════════════════════════════════


def clv(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    new_col: str = "CLV",
    append: bool = True,
) -> pd.DataFrame:
    """Close Location Value.

    ``CLV = ((close - low) - (high - close)) / (high - low) * 100``
    Range: [-100, +100]. +100 = close at high, -100 = close at low.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    result = df.copy()
    result[new_col] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        h = sub[hcol].values
        l = sub[lcol].values
        c = sub[ccol].values
        hl = h - l
        result.loc[idx, new_col] = np.where(hl > 0, ((c - l) - (h - c)) / hl * 100.0, 0.0)

    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════
# Chaikin AD / Accumulation Distribution
# ══════════════════════════════════════════════════════════════════════════


def chaikin_ad(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    new_col: str = "Chaikin_AD",
    append: bool = True,
) -> pd.DataFrame:
    """Chaikin Accumulation Distribution Line.

    ``AD[i] = AD[i-1] + CLV_factor * volume[i]``
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    vcol = _resolve_col(df, "volume", volume_col)
    result = df.copy()
    result[new_col] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        h = sub[hcol].values
        l = sub[lcol].values
        c = sub[ccol].values
        v = sub[vcol].values
        hl = h - l
        clv_factor = np.where(hl > 0, ((c - l) - (h - c)) / hl, 0.0)
        ad = np.cumsum(clv_factor * v)
        ad = np.concatenate([[np.nan], ad[:-1] if len(ad) > 0 else []])
        result.loc[sub.index, new_col] = ad

    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════
# Chaikin Volatility
# ══════════════════════════════════════════════════════════════════════════


def chaikin_volatility(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    n: int = 10,
    n_change: int = 10,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Chaikin Volatility.

    ``ROC of EMA(high - low)``
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    cname = new_col or f"ChaikinVol_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        hl = sub[hcol].values - sub[lcol].values
        ema_hl = roll_ema(hl, n)
        ch_vol = pd.Series(ema_hl).pct_change(periods=n_change).values * 100.0
        result.loc[idx, cname] = ch_vol

    return slim_output(result, cname, append)


# ══════════════════════════════════════════════════════════════════════════
# Williams AD
# ══════════════════════════════════════════════════════════════════════════


def williams_ad(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    new_col: str = "Williams_AD",
    append: bool = True,
) -> pd.DataFrame:
    """Williams Accumulation Distribution.

    Up day: AD += close - true_low. Down day: AD += close - true_high.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    result = df.copy()
    result[new_col] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        h = sub[hcol].values
        l = sub[lcol].values
        c = sub[ccol].values

        ad = np.zeros(len(c), dtype=np.float64)
        cum = 0.0
        for i in range(1, len(c)):
            if np.isnan(c[i]) or np.isnan(c[i - 1]):
                cum = 0.0 if np.isnan(c[i]) else cum
                ad[i] = np.nan
                continue
            if c[i] > c[i - 1]:
                tlow = min(l[i], c[i - 1])
                cum += c[i] - tlow
            elif c[i] < c[i - 1]:
                thigh = max(h[i], c[i - 1])
                cum += c[i] - thigh
            ad[i] = cum
        ad[0] = np.nan
        result.loc[sub.index, new_col] = ad

    return slim_output(result, new_col, append)
