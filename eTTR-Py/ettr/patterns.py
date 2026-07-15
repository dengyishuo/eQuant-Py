"""Chart pattern indicators.

Each function takes a long-format panel DataFrame and returns it
with new indicator columns appended.

Equivalent to eTTR/R/add_ZigZag.R, add_pivots.R, add_SAR.R, etc.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ettr._panel import _resolve_col
from equant.utils.panel import slim_output, validate_panel


# ══════════════════════════════════════════════════════════════════════════
# ZigZag
# ══════════════════════════════════════════════════════════════════════════


def zigzag(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    change: float = 0.1,
    pct: bool = True,
    retrace: bool = False,
    last_extreme: bool = True,
    new_col: str = "ZigZag",
    append: bool = True,
) -> pd.DataFrame:
    """ZigZag — identifies swing highs and lows.

    Parameters
    ----------
    change : float
        Minimum price change to confirm a new swing point.
    pct : bool
        If True, *change* is in percentage; else absolute.
    retrace : bool
        If True, require retracement before confirming.
    last_extreme : bool
        If True, extend the last confirmed extreme to the final bar.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    result = df.copy()
    result[new_col] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        h = sub[hcol].values.astype(np.float64)
        l = sub[lcol].values.astype(np.float64)

        n = len(h)
        zz = np.full(n, np.nan)

        if n < 3:
            result.loc[sub.index, new_col] = zz
            continue

        # Find first valid price
        start = 0
        while start < n and (np.isnan(h[start]) or np.isnan(l[start])):
            start += 1
        if start >= n:
            continue

        # Determine initial direction
        direction = 0  # 0=unknown, 1=up, -1=down
        last_high_idx = start
        last_low_idx = start
        last_high_val = h[start]
        last_low_val = l[start]

        for i in range(start + 1, n):
            if np.isnan(h[i]) or np.isnan(l[i]):
                continue

            threshold = change / 100.0 if pct else change

            if direction == 0:
                # Initialize direction
                if h[i] > last_high_val * (1 + threshold):
                    direction = 1
                    last_low_idx = i - 1 if i > 0 else 0
                    last_low_val = l[last_low_idx]
                    zz[last_low_idx] = last_low_val
                elif l[i] < last_low_val * (1 - threshold):
                    direction = -1
                    last_high_idx = i - 1 if i > 0 else 0
                    last_high_val = h[last_high_idx]
                    zz[last_high_idx] = last_high_val

            elif direction == 1:
                if h[i] > last_high_val:
                    last_high_val = h[i]
                    last_high_idx = i
                if l[i] < last_high_val * (1 - threshold):
                    if not retrace or l[i] < last_low_val:
                        zz[last_high_idx] = last_high_val
                        direction = -1
                        last_low_val = l[i]
                        last_low_idx = i

            elif direction == -1:
                if l[i] < last_low_val:
                    last_low_val = l[i]
                    last_low_idx = i
                if h[i] > last_low_val * (1 + threshold):
                    if not retrace or h[i] > last_high_val:
                        zz[last_low_idx] = last_low_val
                        direction = 1
                        last_high_val = h[i]
                        last_high_idx = i

        # Mark the last extreme
        if direction == 1:
            zz[last_high_idx] = last_high_val
        elif direction == -1:
            zz[last_low_idx] = last_low_val

        if last_extreme and not np.isnan(zz[last_high_idx if direction >= 0 else last_low_idx]):
            zz[-1] = h[-1] if direction >= 0 else l[-1]

        result.loc[sub.index, new_col] = zz

    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════
# Pivots (Floor Pivots)
# ══════════════════════════════════════════════════════════════════════════


def pivots(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    new_col: str = "Pivot",
    append: bool = True,
) -> pd.DataFrame:
    """Floor Pivot Points.

    Produces ``{new_col}_PP``, ``{new_col}_R1``, ``{new_col}_R2``, ``{new_col}_S1``, ``{new_col}_S2``.

    Each day's pivots are computed from the previous day's OHLC.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    out_cols = [f"{new_col}_PP", f"{new_col}_R1", f"{new_col}_R2", f"{new_col}_S1", f"{new_col}_S2"]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        h = sub[hcol].values
        l = sub[lcol].values
        c = sub[ccol].values

        pp = np.full(len(h), np.nan)
        r1 = np.full(len(h), np.nan)
        r2 = np.full(len(h), np.nan)
        s1 = np.full(len(h), np.nan)
        s2 = np.full(len(h), np.nan)

        for i in range(1, len(h)):
            prev_h, prev_l, prev_c = h[i - 1], l[i - 1], c[i - 1]
            if np.isnan(prev_h) or np.isnan(prev_l) or np.isnan(prev_c):
                continue
            pp[i] = (prev_h + prev_l + prev_c) / 3.0
            r1[i] = 2.0 * pp[i] - prev_l
            s1[i] = 2.0 * pp[i] - prev_h
            r2[i] = pp[i] + (prev_h - prev_l)
            s2[i] = pp[i] - (prev_h - prev_l)

        result.loc[sub.index, out_cols[0]] = pp
        result.loc[sub.index, out_cols[1]] = r1
        result.loc[sub.index, out_cols[2]] = r2
        result.loc[sub.index, out_cols[3]] = s1
        result.loc[sub.index, out_cols[4]] = s2

    return slim_output(result, out_cols, append)


# ══════════════════════════════════════════════════════════════════════════
# SAR — Parabolic Stop and Reverse
# ══════════════════════════════════════════════════════════════════════════


def sar(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    accel_init: float = 0.02,
    accel_max: float = 0.20,
    new_col: str = "SAR",
    append: bool = True,
) -> pd.DataFrame:
    """Parabolic SAR.

    Standard Wilder implementation with acceleration factor.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    result = df.copy()
    result[new_col] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        h = sub[hcol].values.astype(np.float64)
        l = sub[lcol].values.astype(np.float64)

        n = len(h)
        sar_vals = np.full(n, np.nan)

        if n < 2:
            result.loc[sub.index, new_col] = sar_vals
            continue

        # Initialize: start long
        long = True
        af = accel_init
        ep = h[0]  # extreme point
        sar_val = l[0] - (h[0] - l[0]) * 0.1  # initial SAR below first low

        for i in range(1, n):
            if np.isnan(h[i]) or np.isnan(l[i]):
                continue

            sar_val = sar_val + af * (ep - sar_val)

            if long:
                sar_val = min(sar_val, l[i - 1])
                if i > 1:
                    sar_val = min(sar_val, l[i - 2])

                if l[i] < sar_val:
                    # Reverse to short
                    long = False
                    sar_val = ep
                    ep = l[i]
                    af = accel_init
                else:
                    if h[i] > ep:
                        ep = h[i]
                        af = min(af + accel_init, accel_max)
            else:
                sar_val = max(sar_val, h[i - 1])
                if i > 1:
                    sar_val = max(sar_val, h[i - 2])

                if h[i] > sar_val:
                    # Reverse to long
                    long = True
                    sar_val = ep
                    ep = h[i]
                    af = accel_init
                else:
                    if l[i] < ep:
                        ep = l[i]
                        af = min(af + accel_init, accel_max)

            sar_vals[i] = sar_val

        result.loc[sub.index, new_col] = sar_vals

    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════
# SNR — Support and Resistance
# ══════════════════════════════════════════════════════════════════════════


def snr(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    n: int = 20,
    new_col: str = "SNR",
    append: bool = True,
) -> pd.DataFrame:
    """DeMark-like Support and Resistance levels.

    Produces ``{new_col}_R``, ``{new_col}_S``, ``{new_col}_mid``.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    out_cols = [f"{new_col}_R", f"{new_col}_S", f"{new_col}_mid"]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        h = sub[hcol].values
        l = sub[lcol].values
        c = sub[ccol].values

        r_vals = np.full(len(h), np.nan)
        s_vals = np.full(len(h), np.nan)

        for i in range(1, len(h)):
            if np.isnan(c[i]) or np.isnan(c[i - 1]):
                continue
            if c[i] > c[i - 1]:
                r_vals[i] = (h[i] + c[i]) / 2.0
                s_vals[i] = (l[i] + c[i]) / 2.0
            else:
                r_vals[i] = (h[i - 1] + c[i - 1]) / 2.0
                s_vals[i] = (l[i - 1] + c[i - 1]) / 2.0

        result.loc[sub.index, out_cols[0]] = r_vals
        result.loc[sub.index, out_cols[1]] = s_vals
        result.loc[sub.index, out_cols[2]] = (r_vals + s_vals) / 2.0

    return slim_output(result, out_cols, append)
