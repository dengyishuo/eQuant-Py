"""Technical indicator overlays for candlestick charts.

These functions compute overlays that are drawn directly on the price chart
(e.g., moving averages, Bollinger Bands, SAR, Donchian, Keltner channels).
They return pandas Series/DataFrames suitable for mplfinance addplot usage.
"""

from typing import Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def sma(data: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return data.rolling(period).mean()


def ema(data: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return data.ewm(span=period, adjust=False).mean()


def wma(data: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    return data.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum())


def compute_ma(
    data: pd.DataFrame,
    periods: Union[int, list[int]],
    ma_type: str = "sma",
    price_col: str = "close",
) -> dict[int, pd.Series]:
    """Compute one or more moving averages.

    Parameters
    ----------
    data : pd.DataFrame
        OHLC data.  Must contain `price_col`.
    periods : int or list of int
        Moving average period(s).
    ma_type : str
        Type of MA: "sma", "ema", "wma".
    price_col : str
        Which price column to use (default "close").

    Returns
    -------
    dict[int, pd.Series]
        Mapping of period -> MA series.
    """
    if isinstance(periods, int):
        periods = [periods]

    ma_funcs = {"sma": sma, "ema": ema, "wma": wma}
    func = ma_funcs.get(ma_type.lower(), sma)

    price = data[price_col]
    result = {}
    for p in periods:
        result[p] = func(price, p)
    return result


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def _compute_bbands(
    data: pd.DataFrame,
    period: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    price_col: str = "close",
) -> dict[str, pd.Series]:
    """Compute Bollinger Bands.

    Returns
    -------
    dict with keys 'upper', 'middle', 'lower'.
    """
    price = data[price_col]
    middle = price.rolling(period).mean()
    std = price.rolling(period).std()
    return {
        "upper": middle + nbdevup * std,
        "middle": middle,
        "lower": middle - nbdevdn * std,
    }


# ---------------------------------------------------------------------------
# Parabolic SAR
# ---------------------------------------------------------------------------

def _compute_sar(
    data: pd.DataFrame,
    accel_start: float = 0.02,
    accel_max: float = 0.2,
) -> pd.Series:
    """Compute Parabolic SAR.

    Implementation follows Wilder's original algorithm.

    Returns
    -------
    pd.Series
        SAR values. NaN where not applicable (first few bars).
    """
    high = data["high"].values
    low = data["low"].values
    n = len(high)

    sar = np.full(n, np.nan)

    # Determine initial trend from first two bars
    if n < 2:
        return pd.Series(sar, index=data.index)

    # Initial values
    trend_up = True
    ep = high[0]  # extreme point
    sar[0] = low[0]
    sar[1] = sar[0]
    af = accel_start

    for i in range(1, n):
        prev_sar = sar[i - 1]

        if trend_up:
            sar[i] = prev_sar + af * (ep - prev_sar)
            # Ensure SAR is below the lowest low of the two most recent bars
            if i >= 1:
                sar[i] = min(sar[i], low[max(0, i - 1)], low[i])

            if high[i] > ep:
                ep = high[i]
                af = min(af + accel_start, accel_max)

            # Reversal check
            if low[i] < sar[i]:
                trend_up = False
                sar[i] = ep  # SAR becomes the EP for the new downtrend
                ep = low[i]
                af = accel_start
        else:
            sar[i] = prev_sar - af * (prev_sar - ep)
            if i >= 1:
                sar[i] = max(sar[i], high[max(0, i - 1)], high[i])

            if low[i] < ep:
                ep = low[i]
                af = min(af + accel_start, accel_max)

            # Reversal check
            if high[i] > sar[i]:
                trend_up = True
                sar[i] = ep
                ep = high[i]
                af = accel_start

    return pd.Series(sar, index=data.index)


# ---------------------------------------------------------------------------
# Donchian Channel
# ---------------------------------------------------------------------------

def _compute_donchian(
    data: pd.DataFrame,
    period: int = 20,
) -> dict[str, pd.Series]:
    """Compute Donchian channel.

    Returns
    -------
    dict with keys 'upper', 'mid', 'lower'.
    """
    high = data["high"]
    low = data["low"]
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    mid = (upper + lower) / 2
    return {"upper": upper, "mid": mid, "lower": lower}


# ---------------------------------------------------------------------------
# Keltner Channel
# ---------------------------------------------------------------------------

def _compute_keltner(
    data: pd.DataFrame,
    period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> dict[str, pd.Series]:
    """Compute Keltner channel using EMA of typical price and ATR.

    Returns
    -------
    dict with keys 'upper', 'mid', 'lower'.
    """
    from .indicators import _compute_atr

    typical = (data["high"] + data["low"] + data["close"]) / 3
    mid = typical.ewm(span=period, adjust=False).mean()
    atr_val = _compute_atr(data, atr_period)
    return {
        "upper": mid + multiplier * atr_val,
        "mid": mid,
        "lower": mid - multiplier * atr_val,
    }
