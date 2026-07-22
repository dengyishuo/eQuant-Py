"""Standalone technical indicator computation for panel charts.

These functions compute indicators that are typically displayed in separate
panels below the price chart (MACD, RSI, ADX, ATR, Stochastic, CCI, OBV, etc.).
They return pandas Series/DataFrames suitable for mplfinance addplot panels.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def _compute_macd(
    data: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "close",
) -> dict[str, pd.Series]:
    """Compute MACD.

    Returns dict with keys 'macd', 'signal', 'histogram'.
    """
    price = data[price_col]
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return {"macd": macd, "signal": signal_line, "histogram": histogram}


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def _compute_rsi(
    data: pd.DataFrame,
    period: int = 14,
    price_col: str = "close",
) -> pd.Series:
    """Compute RSI (Relative Strength Index) using Wilder's smoothing.

    Returns pd.Series of RSI values.
    """
    price = data[price_col]
    delta = price.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def _compute_atr(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """Compute ATR (Average True Range) using Wilder's smoothing.

    Returns pd.Series of ATR values.
    """
    high = data["high"]
    low = data["low"]
    close = data["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    return atr


# ---------------------------------------------------------------------------
# ADX / DMI
# ---------------------------------------------------------------------------

def _compute_adx(
    data: pd.DataFrame,
    period: int = 14,
) -> dict[str, pd.Series]:
    """Compute ADX (Average Directional Index) and DMI lines.

    Returns dict with keys 'adx', '+di', '-di'.
    """
    high = data["high"]
    low = data["low"]
    close = data["close"]

    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > 0) & (up_move > down_move.fillna(0)), 0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move.fillna(0)), 0)

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return {"adx": adx, "+di": plus_di, "-di": minus_di}


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------

def _compute_stoch(
    data: pd.DataFrame,
    k_period: int = 14,
    k_slow: int = 3,
    d_period: int = 3,
) -> dict[str, pd.Series]:
    """Compute Stochastic Oscillator (%K and %D).

    Returns dict with keys '%K', '%D'.
    """
    high = data["high"]
    low = data["low"]
    close = data["close"]

    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()

    fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    slow_k = fast_k.rolling(k_slow).mean()
    slow_d = slow_k.rolling(d_period).mean()

    return {"%K": slow_k, "%D": slow_d}


# ---------------------------------------------------------------------------
# CCI
# ---------------------------------------------------------------------------

def _compute_cci(
    data: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    """Compute CCI (Commodity Channel Index).

    Returns pd.Series.
    """
    tp = (data["high"] + data["low"] + data["close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci


# ---------------------------------------------------------------------------
# OBV
# ---------------------------------------------------------------------------

def _compute_obv(data: pd.DataFrame) -> pd.Series:
    """Compute OBV (On-Balance Volume).

    Returns pd.Series.
    """
    close = data["close"]
    volume = data.get("volume", pd.Series(1, index=data.index))

    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume).cumsum()
    return obv


# ---------------------------------------------------------------------------
# CMF (Chaikin Money Flow)
# ---------------------------------------------------------------------------

def _compute_cmf(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute CMF (Chaikin Money Flow).

    Returns pd.Series.
    """
    high = data["high"]
    low = data["low"]
    close = data["close"]
    volume = data.get("volume", pd.Series(1, index=data.index))

    hl_range = high - low
    hl_range = hl_range.replace(0, np.nan)
    mf_multiplier = ((close - low) - (high - close)) / hl_range
    mf_volume = mf_multiplier * volume
    cmf = mf_volume.rolling(period).sum() / volume.rolling(period).sum()
    return cmf


# ---------------------------------------------------------------------------
# MFI (Money Flow Index)
# ---------------------------------------------------------------------------

def _compute_mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute MFI (Money Flow Index).

    Returns pd.Series.
    """
    tp = (data["high"] + data["low"] + data["close"]) / 3
    volume = data.get("volume", pd.Series(1, index=data.index))

    raw_mf = tp * volume
    mf_direction = np.sign(tp.diff().fillna(0))

    pos_flow = raw_mf.where(mf_direction > 0, 0).rolling(period).sum()
    neg_flow = raw_mf.where(mf_direction < 0, 0).rolling(period).sum()

    mfr = pos_flow / neg_flow.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    return mfi


# ---------------------------------------------------------------------------
# WPR (Williams %R)
# ---------------------------------------------------------------------------

def _compute_wpr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Williams %R.

    Returns pd.Series.
    """
    high = data["high"]
    low = data["low"]
    close = data["close"]

    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()

    wpr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wpr


# ---------------------------------------------------------------------------
# KDJ
# ---------------------------------------------------------------------------

def _compute_kdj(
    data: pd.DataFrame,
    n: int = 9,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> dict[str, pd.Series]:
    """Compute KDJ indicator.

    Returns dict with keys 'K', 'D', 'J'.
    """
    high = data["high"]
    low = data["low"]
    close = data["close"]

    lowest_low = low.rolling(n).min()
    highest_high = high.rolling(n).max()

    rsv = 100 * (close - lowest_low) / (highest_high - lowest_low)

    k = rsv.ewm(alpha=1 / k_smooth, adjust=False).mean()
    d = k.ewm(alpha=1 / d_smooth, adjust=False).mean()
    j = 3 * k - 2 * d

    return {"K": k, "D": d, "J": j}
