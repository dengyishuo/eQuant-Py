"""Volatility indicators and estimators.

Each function takes a long-format panel DataFrame and returns it
with new indicator columns appended.

Equivalent to eTTR/R/add_ATR.R, add_BBands.R, volatility.R, etc.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ettr._panel import _resolve_col
from ettr._rolling import roll_ema, wilder_sum
from ettr.trend import _true_range
from equant.utils.panel import slim_output, validate_panel


# ══════════════════════════════════════════════════════════════════════════
# ATR — Average True Range
# ══════════════════════════════════════════════════════════════════════════


def atr(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    n: int = 14,
    wilder: bool = True,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Average True Range.

    Parameters
    ----------
    wilder : bool
        If True (default), uses Wilder's smoothing for the average.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    cname = new_col or f"ATR_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        tr = _true_range(sub[hcol].values, sub[lcol].values, sub[ccol].values)

        if wilder:
            result.loc[idx, cname] = wilder_sum(tr, n)
        else:
            result.loc[idx, cname] = pd.Series(tr).rolling(window=n, min_periods=n).mean().values

    return slim_output(result, cname, append)


def tr(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    new_col: str = "TR",
    append: bool = True,
) -> pd.DataFrame:
    """True Range."""
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    result = df.copy()
    result[new_col] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        result.loc[idx, new_col] = _true_range(
            sub[hcol].values, sub[lcol].values, sub[ccol].values
        )

    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════
# Bollinger Bands
# ══════════════════════════════════════════════════════════════════════════


def bollinger(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 20,
    sd: float = 2.0,
    new_col: str = "BB",
    append: bool = True,
) -> pd.DataFrame:
    """Bollinger Bands.

    Produces ``{new_col}_middle``, ``{new_col}_upper``, ``{new_col}_lower``,
    ``{new_col}_pctB``, ``{new_col}_width``.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    out_cols = [
        f"{new_col}_middle", f"{new_col}_upper", f"{new_col}_lower",
        f"{new_col}_pctB", f"{new_col}_width",
    ]
    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        middle = pd.Series(vals).rolling(window=n, min_periods=n).mean().values
        std = pd.Series(vals).rolling(window=n, min_periods=n).std(ddof=1).values

        upper = middle + sd * std
        lower = middle - sd * std
        spread = upper - lower
        pct_b = np.where(spread > 0, (vals - lower) / spread, 0.5)
        width = np.where(middle > 0, spread / middle, np.nan)

        result.loc[idx, out_cols[0]] = middle
        result.loc[idx, out_cols[1]] = upper
        result.loc[idx, out_cols[2]] = lower
        result.loc[idx, out_cols[3]] = pct_b
        result.loc[idx, out_cols[4]] = width

    return slim_output(result, out_cols, append)


# ══════════════════════════════════════════════════════════════════════════
# Keltner Channels
# ══════════════════════════════════════════════════════════════════════════


def keltner(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    n_ema: int = 20,
    n_atr: int = 14,
    multiplier: float = 2.0,
    new_col: str = "Keltner",
    append: bool = True,
) -> pd.DataFrame:
    """Keltner Channels.

    Produces ``{new_col}_middle``, ``{new_col}_upper``, ``{new_col}_lower``.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    out_cols = [f"{new_col}_middle", f"{new_col}_upper", f"{new_col}_lower"]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        price = sub[ccol].values.astype(np.float64)
        tr_vals = _true_range(sub[hcol].values, sub[lcol].values, sub[ccol].values)

        middle = roll_ema(price, n_ema)
        atr_vals = roll_ema(tr_vals, n_atr)

        result.loc[idx, out_cols[0]] = middle
        result.loc[idx, out_cols[1]] = middle + multiplier * atr_vals
        result.loc[idx, out_cols[2]] = middle - multiplier * atr_vals

    return slim_output(result, out_cols, append)


# ══════════════════════════════════════════════════════════════════════════
# Donchian Channel
# ══════════════════════════════════════════════════════════════════════════


def donchian(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    n: int = 20,
    new_col: str = "DC",
    append: bool = True,
) -> pd.DataFrame:
    """Donchian Channel.

    Produces ``{new_col}_upper``, ``{new_col}_lower``, ``{new_col}_middle``.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    out_cols = [f"{new_col}_upper", f"{new_col}_lower", f"{new_col}_middle"]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        high = sub[hcol].values
        low = sub[lcol].values
        upper = pd.Series(high).rolling(window=n, min_periods=n).max().values
        lower = pd.Series(low).rolling(window=n, min_periods=n).min().values

        result.loc[idx, out_cols[0]] = upper
        result.loc[idx, out_cols[1]] = lower
        result.loc[idx, out_cols[2]] = (upper + lower) / 2.0

    return slim_output(result, out_cols, append)


# ══════════════════════════════════════════════════════════════════════════
# PBands — Percentage Bands
# ══════════════════════════════════════════════════════════════════════════


def pbands(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 20,
    sd: float = 2.0,
    new_col: str = "PB",
    append: bool = True,
) -> pd.DataFrame:
    """Percentage Bands (Bollinger Bands as percentage of middle).

    Produces ``{new_col}_upper``, ``{new_col}_lower``.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    out_cols = [f"{new_col}_upper", f"{new_col}_lower"]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        middle = pd.Series(vals).rolling(window=n, min_periods=n).mean().values
        std = pd.Series(vals).rolling(window=n, min_periods=n).std(ddof=1).values
        pct = np.where(middle > 0, sd * std / middle * 100.0, np.nan)

        result.loc[idx, out_cols[0]] = pct
        result.loc[idx, out_cols[1]] = -pct

    return slim_output(result, out_cols, append)


# ══════════════════════════════════════════════════════════════════════════
# Volatility Estimators (6 types)
# ══════════════════════════════════════════════════════════════════════════


def volatility(
    df: pd.DataFrame,
    open_col: Optional[str] = None,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    n: int = 10,
    calc: str = "close",
    N: int = 260,
    mean0: bool = False,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Historical volatility estimation.

    Parameters
    ----------
    n : int
        Rolling window length.
    calc : str
        One of: ``"close"``, ``"garman.klass"``, ``"parkinson"``,
        ``"rogers.satchell"``, ``"gk.yz"``, ``"yang.zhang"``.
    N : int
        Annualization factor (260 = daily → annual).
    mean0 : bool
        If True, use mean=0 for close-to-close. Only for ``calc="close"``.
    """
    validate_panel(df)
    ocol = _resolve_col(df, "open", open_col)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    cname = new_col or f"Vol_{calc}_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        o = sub[ocol].values.astype(np.float64)
        h = sub[hcol].values.astype(np.float64)
        low = sub[lcol].values.astype(np.float64)
        c = sub[ccol].values.astype(np.float64)

        if calc == "close":
            r = np.full(len(c), np.nan)
            r[1:] = np.log(c[1:] / c[:-1])
            if mean0:
                vol_vals = np.sqrt(N) * np.sqrt(
                    pd.Series(r**2).rolling(n - 1, min_periods=n - 1).sum().values / (n - 2)
                )
            else:
                vol_vals = np.sqrt(N) * pd.Series(r).rolling(n - 1, min_periods=n - 1).std(ddof=1).values
            result.loc[idx, cname] = vol_vals

        elif calc == "garman.klass":
            term = 0.5 * np.log(h / (low + 1e-15)) ** 2 - (2 * np.log(2) - 1) * np.log(c / (o + 1e-15)) ** 2
            vol_vals = np.sqrt(N / n * pd.Series(term).rolling(n, min_periods=n).sum().values)
            result.loc[idx, cname] = vol_vals

        elif calc == "parkinson":
            term = np.log(h / (low + 1e-15)) ** 2
            vol_vals = np.sqrt(N / (4 * n * np.log(2)) * pd.Series(term).rolling(n, min_periods=n).sum().values)
            result.loc[idx, cname] = vol_vals

        elif calc == "rogers.satchell":
            term = np.log(h / (c + 1e-15)) * np.log(h / (o + 1e-15)) + np.log(low / (c + 1e-15)) * np.log(low / (o + 1e-15))
            vol_vals = np.sqrt(N / n * pd.Series(term).rolling(n, min_periods=n).sum().values)
            result.loc[idx, cname] = vol_vals

        elif calc == "gk.yz":
            cl1 = np.concatenate([[np.nan], c[:-1]])
            term = np.log(o / (cl1 + 1e-15)) ** 2 + 0.5 * np.log(h / (low + 1e-15)) ** 2 - (2 * np.log(2) - 1) * np.log(c / (o + 1e-15)) ** 2
            vol_vals = np.sqrt(N / n * pd.Series(term).rolling(n, min_periods=n).sum().values)
            result.loc[idx, cname] = vol_vals

        elif calc == "yang.zhang":
            cl1 = np.concatenate([[np.nan], c[:-1]])
            alpha = 1.34
            k = (alpha - 1) / (alpha + (n + 1) / (n - 1))

            s2o = N * pd.Series(np.log(o / (cl1 + 1e-15))).rolling(n, min_periods=n).var(ddof=1).values
            s2c = N * pd.Series(np.log(c / (o + 1e-15))).rolling(n, min_periods=n).var(ddof=1).values

            rs_term = np.log(h / (c + 1e-15)) * np.log(h / (o + 1e-15)) + np.log(low / (c + 1e-15)) * np.log(low / (o + 1e-15))
            s2rs = N / n * pd.Series(rs_term).rolling(n, min_periods=n).sum().values

            vol_vals = np.sqrt(s2o + k * s2c + (1 - k) * (s2rs))
            result.loc[idx, cname] = vol_vals

    return slim_output(result, cname, append)
