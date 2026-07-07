"""Trend-following technical indicators.

Each function takes a long-format panel DataFrame and returns it
with new indicator columns appended.

Equivalent to eTTR/R/add_SMA.R, add_EMA.R, add_MACD.R, add_ADX.R, etc.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from ettr._panel import _resolve_col, apply_by_code
from ettr._rolling import (
    roll_ema,
    roll_evwma,
    roll_wma,
    roll_zlema,
    wilder_sum,
)
from equant.utils.panel import slim_output, sort_panel, validate_panel


def _sma_1d(x: pd.Series, n: int) -> pd.Series:
    """Simple moving average for a single asset series."""
    return x.rolling(window=n, min_periods=n).mean()


def sma(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: Union[int, Sequence[int]] = 20,
    new_col: str = "SMA",
    append: bool = True,
) -> pd.DataFrame:
    """Simple Moving Average.

    Parameters
    ----------
    df : DataFrame
        Long-format panel data.
    close_col : str, optional
        Price column name. Defaults to ``"close"``.
    n : int or sequence of int
        Lookback period(s). Multiple values produce multiple columns.
    new_col : str
        Prefix for output columns (e.g., ``SMA_20``).
    append : bool
        If False, return only id columns plus new columns.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)

    ns = [n] if isinstance(n, int) else list(n)
    result = df.copy()

    for period in ns:
        cname = f"{new_col}_{period}"
        result[cname] = np.nan

        for _code, idx in result.groupby("code", sort=False).groups.items():
            result.loc[idx, cname] = _sma_1d(result.loc[idx, col], period).values

    return slim_output(result, [f"{new_col}_{p}" for p in ns], append)


def ema(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: Union[int, Sequence[int]] = 20,
    wilder: bool = False,
    new_col: str = "EMA",
    append: bool = True,
) -> pd.DataFrame:
    """Exponential Moving Average.

    Parameters
    ----------
    n : int or sequence of int
        Lookback period(s).
    wilder : bool
        Use Wilder's smoothing ratio (``1/n`` instead of ``2/(n+1)``).
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
            result.loc[idx, cname] = roll_ema(vals, period, wilder)

    return slim_output(result, [f"{new_col}_{p}" for p in ns], append)


def dema(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 10,
    v: float = 1.0,
    wilder: bool = False,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Double Exponential Moving Average.

    ``DEMA = 2 * EMA(x) - EMA(EMA(x))``, blended with *v*.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or f"DEMA_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        ema1 = roll_ema(vals, n, wilder)
        ema2 = roll_ema(ema1, n, wilder)
        dema_vals = (1 + v) * ema1 - v * ema2
        result.loc[idx, cname] = dema_vals

    return slim_output(result, cname, append)


def wma(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 10,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Weighted Moving Average (linearly increasing weights)."""
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or f"WMA_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        result.loc[idx, cname] = roll_wma(vals, n)

    return slim_output(result, cname, append)


def hma(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 20,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Hull Moving Average.

    ``HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))``
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or f"HMA_{n}"
    n2 = max(1, n // 2)
    n_sqrt = max(1, int(np.sqrt(n)))

    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        wma_half = roll_wma(vals, n2)
        wma_full = roll_wma(vals, n)
        raw = 2.0 * wma_half - wma_full
        mask = ~np.isnan(raw)
        raw_clean = np.where(mask, raw, 0.0)
        hma_vals = roll_wma(raw_clean, n_sqrt)
        hma_vals[~mask] = np.nan
        result.loc[idx, cname] = hma_vals

    return slim_output(result, cname, append)


def zlema(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 20,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Zero-Lag Exponential Moving Average."""
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or f"ZLEMA_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        result.loc[idx, cname] = roll_zlema(vals, n)

    return slim_output(result, cname, append)


def alma(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 9,
    offset: float = 0.85,
    sigma: float = 6.0,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Arnaud Legoux Moving Average (Gaussian-weighted).

    Parameters
    ----------
    n : int
        Window length.
    offset : float
        Center of Gaussian weight (0-1). Higher = more responsive.
    sigma : float
        Width of Gaussian bell. Lower = less smoothing.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or f"ALMA_{n}"

    # Pre-compute Gaussian weights
    m = int(np.floor(offset * (n - 1)))
    s = n / sigma
    wts = np.exp(-((np.arange(n) - m) ** 2) / (2 * s * s))
    wts /= wts.sum()

    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        alma_vals = roll_wma(vals, n, weights=wts)
        result.loc[idx, cname] = alma_vals

    return slim_output(result, cname, append)


def evwma(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    n: int = 10,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Elastic Volume-Weighted Moving Average."""
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    vol_col = _resolve_col(df, "volume", volume_col)
    cname = new_col or f"EVWMA_{n}"

    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        price = result.loc[idx, col].values.astype(np.float64)
        volume = result.loc[idx, vol_col].values.astype(np.float64)
        result.loc[idx, cname] = roll_evwma(price, volume, n)

    return slim_output(result, cname, append)


def vwma(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    n: int = 20,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Volume-Weighted Moving Average.

    ``VWMA = sum(price * volume) / sum(volume)`` over *n* periods.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    vol_col = _resolve_col(df, "volume", volume_col)
    cname = new_col or f"VWMA_{n}"

    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        pv = sub[col] * sub[vol_col]
        num = pv.rolling(window=n, min_periods=n).sum()
        denom = sub[vol_col].rolling(window=n, min_periods=n).sum()
        result.loc[idx, cname] = (num / denom.replace(0, np.nan)).values

    return slim_output(result, cname, append)


def macd(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n_fast: int = 12,
    n_slow: int = 26,
    n_signal: int = 9,
    wilder: bool = False,
    new_col: str = "MACD",
    append: bool = True,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Produces three columns: ``{new_col}``, ``{new_col}_signal``, ``{new_col}_hist``.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    out_cols = [new_col, f"{new_col}_signal", f"{new_col}_hist"]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        fast = roll_ema(vals, n_fast, wilder)
        slow = roll_ema(vals, n_slow, wilder)
        d = fast - slow
        sig = roll_ema(d, n_signal, wilder)
        result.loc[idx, new_col] = d
        result.loc[idx, f"{new_col}_signal"] = sig
        result.loc[idx, f"{new_col}_hist"] = d - sig

    return slim_output(result, out_cols, append)


def adx(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    n: int = 14,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Average Directional Movement Index.

    Produces: ``ADX_{n}``, ``DIp_{n}``, ``DIn_{n}``, ``DX_{n}``.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    ccol = _resolve_col(df, "close", close_col)
    prefix = new_col or ""
    adx_c = f"{prefix}ADX_{n}" if prefix else f"ADX_{n}"
    dip_c = f"{prefix}DIp_{n}" if prefix else f"DIp_{n}"
    din_c = f"{prefix}DIn_{n}" if prefix else f"DIn_{n}"
    dx_c = f"{prefix}DX_{n}" if prefix else f"DX_{n}"
    out_cols = [adx_c, dip_c, din_c, dx_c]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx]
        high = sub[hcol].values.astype(np.float64)
        low = sub[lcol].values.astype(np.float64)
        cl = sub[ccol].values.astype(np.float64)

        tr_arr = _true_range(high, low, cl)
        atr = wilder_sum(tr_arr, n)

        up_move = np.empty(len(high))
        down_move = np.empty(len(high))
        for i in range(1, len(high)):
            up_move[i] = high[i] - high[i - 1] if high[i] > high[i - 1] else 0.0
            down_move[i] = low[i - 1] - low[i] if low[i] < low[i - 1] else 0.0
        up_move[0] = np.nan
        down_move[0] = np.nan

        sm_up = wilder_sum(np.where(up_move > down_move, up_move, 0.0), n)
        sm_down = wilder_sum(np.where(down_move > up_move, down_move, 0.0), n)

        dip = 100.0 * sm_up / atr
        din = 100.0 * sm_down / atr
        dx = 100.0 * np.abs(dip - din) / (dip + din + 1e-15)

        adx_vals = roll_ema(dx, n, wilder=True)
        result.loc[idx, adx_c] = adx_vals
        result.loc[idx, dip_c] = dip
        result.loc[idx, din_c] = din
        result.loc[idx, dx_c] = dx

    return slim_output(result, out_cols, append)


def _true_range(high, low, close):
    """Vectorized true range."""
    tr = np.empty(len(high))
    tr[0] = np.nan
    for i in range(1, len(high)):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i - 1]):
            tr[i] = np.nan
        else:
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
    return tr


def gmma(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    new_col: str = "GMMA",
    append: bool = True,
) -> pd.DataFrame:
    """Guppy Multiple Moving Average.

    Produces 12 columns (6 short-term + 6 long-term EMAs)::

        {new_col}_short_3, 5, 8, 10, 12, 15
        {new_col}_long_30, 35, 40, 45, 50, 60
    """
    short_periods = [3, 5, 8, 10, 12, 15]
    long_periods = [30, 35, 40, 45, 50, 60]
    all_periods = short_periods + long_periods
    labels = [f"{new_col}_short_{p}" if p < 30 else f"{new_col}_long_{p}" for p in all_periods]

    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    result = df.copy()
    for label in labels:
        result[label] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        for period, label in zip(all_periods, labels):
            result.loc[idx, label] = roll_ema(vals, period)

    return slim_output(result, labels, append)


def tdi(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 20,
    multiple: int = 2,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Trend Detection Index.

    Produces ``TDI_{n}`` and ``DI_{n}`` columns.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    prefix = new_col or ""
    tdi_c = f"{prefix}TDI_{n}" if prefix else f"TDI_{n}"
    di_c = f"{prefix}DI_{n}" if prefix else f"DI_{n}"
    out_cols = [tdi_c, di_c]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)

        # Momentum
        mom = np.full(len(vals), np.nan)
        mom[n:] = vals[n:] - vals[:-n]
        mom[np.isnan(mom)] = 0.0

        # Direction Indicator = sum of momentum over n
        di = pd.Series(mom).rolling(window=n, min_periods=n).sum().values
        abs_di = np.abs(di)

        # TDI
        n2 = min(n * multiple, len(vals))
        n2 = max(n2, 1)
        n1 = max(n, 1)

        abs_mom = np.abs(mom)
        abs_mom_2n = pd.Series(abs_mom).rolling(window=n2, min_periods=n2).sum().values
        abs_mom_1n = pd.Series(abs_mom).rolling(window=n1, min_periods=n1).sum().values
        tdi_vals = abs_di - (abs_mom_2n - abs_mom_1n)

        result.loc[idx, tdi_c] = tdi_vals
        result.loc[idx, di_c] = di

    return slim_output(result, out_cols, append)


def trix(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 15,
    n_signal: int = 9,
    wilder: bool = False,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Triple-smoothed Exponential Moving Average Oscillator.

    Produces ``TRIX_{n}`` and ``TRIX_{n}_signal``.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    prefix = new_col or ""
    trix_c = f"{prefix}TRIX_{n}" if prefix else f"TRIX_{n}"
    sig_c = f"{prefix}TRIX_{n}_signal" if prefix else f"TRIX_{n}_signal"
    out_cols = [trix_c, sig_c]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        e1 = roll_ema(vals, n, wilder)
        e2 = roll_ema(e1, n, wilder)
        e3 = roll_ema(e2, n, wilder)

        # TRIX = percent change of triple-smoothed
        trix_vals = np.full(len(vals), np.nan)
        mask = ~np.isnan(e3)
        trix_vals[1:] = np.where(mask[1:] & mask[:-1], (e3[1:] - e3[:-1]) / np.abs(e3[:-1] + 1e-15) * 100.0, np.nan)

        sig = roll_ema(trix_vals, n_signal, wilder)

        result.loc[idx, trix_c] = trix_vals
        result.loc[idx, sig_c] = sig

    return slim_output(result, out_cols, append)


def dpo(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 20,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Detrended Price Oscillator.

    ``DPO = close - SMA(close, n)``, shifted back ``n/2 + 1`` periods.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or f"DPO_{n}"
    shift = n // 2 + 1

    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        sma_vals = pd.Series(vals).rolling(window=n, min_periods=n).mean().values
        dpo_vals = vals - sma_vals
        # Shift back
        shifted = np.roll(dpo_vals, -shift)
        shifted[-shift:] = np.nan
        result.loc[idx, cname] = shifted

    return slim_output(result, cname, append)


def vhf(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 28,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Vertical Horizontal Filter.

    Measures trend strength: close to 1 = trending, close to 0 = choppy.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or f"VHF_{n}"

    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        hh = pd.Series(vals).rolling(window=n, min_periods=n).max().values
        ll = pd.Series(vals).rolling(window=n, min_periods=n).min().values
        numerator = hh - ll
        denominator = pd.Series(np.abs(np.diff(vals, prepend=np.nan))).rolling(window=n, min_periods=n).sum().values
        result.loc[idx, cname] = np.where(denominator > 0, numerator / denominator, np.nan)

    return slim_output(result, cname, append)


def kst(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Know Sure Thing (KST) oscillator.

    Uses the classic configuration: ROC periods (10, 15, 20, 30) with
    SMA smoothing (10, 10, 10, 15) and weights (1, 2, 3, 4).

    Produces: ``KST`` and ``KST_signal``.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    out_cols = ["KST", "KST_signal"]

    roc_periods = [10, 15, 20, 30]
    sma_periods = [10, 10, 10, 15]
    weights = [1, 2, 3, 4]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)

        kst = np.zeros(len(vals))
        for rp, sp, w in zip(roc_periods, sma_periods, weights):
            roc = np.full(len(vals), np.nan)
            roc[rp:] = (vals[rp:] - vals[:-rp]) / np.abs(vals[:-rp] + 1e-15) * 100.0
            roc_sma = pd.Series(roc).rolling(window=sp, min_periods=sp).mean().values
            kst += w * np.nan_to_num(roc_sma, nan=0.0)

        kst_sig = pd.Series(kst).rolling(window=9, min_periods=9).mean().values

        result.loc[idx, "KST"] = np.where(kst != 0, kst, np.nan)
        result.loc[idx, "KST_signal"] = kst_sig

    return slim_output(result, out_cols, append)


def po_(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n_fast: int = 12,
    n_slow: int = 26,
    new_col: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Price Oscillator.

    ``PO = 100 * (EMA_fast - EMA_slow) / EMA_slow``
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = new_col or "PO"

    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        ema_fast = roll_ema(vals, n_fast)
        ema_slow = roll_ema(vals, n_slow)
        po_vals = 100.0 * (ema_fast - ema_slow) / (ema_slow + 1e-15)
        result.loc[idx, cname] = po_vals

    return slim_output(result, cname, append)
