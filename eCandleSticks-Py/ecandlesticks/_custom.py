"""
R-exclusive patterns and N-bar parameterised patterns.

These are implemented directly in pandas/numpy because TA-Lib has no
equivalent CDL function for them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import _ohlc, _rolling_mean


# ─── helpers ─────────────────────────────────────────────────────────────────

def _rolling_median(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).median()


def _candle_length(h: pd.Series, l: pd.Series) -> pd.Series:
    """Relative candle length: (H-L) / midpoint."""
    mid = (h + l) / 2
    return (h - l) / mid.replace(0, np.nan)


def _candle_body_length(o: pd.Series, c: pd.Series) -> pd.Series:
    """Relative candle body length: |C-O| / midpoint."""
    mid = (o + c) / 2
    return (c - o).abs() / mid.replace(0, np.nan)


# ─── 1-bar patterns (R-only) ─────────────────────────────────────────────────

def add_long_candle(df: pd.DataFrame, n: int = 20,
                    threshold: float = 1.0) -> pd.DataFrame:
    """Long White / Black Candle — candle length >= threshold × rolling median.

    Parameters
    ----------
    n : int
        Lookback periods for rolling median. Default 20.
    threshold : float
        Multiplier applied to the median. Default 1.0.

    Returns
    -------
    DataFrame with columns: LongWhiteCandle, LongBlackCandle (bool)
    """
    o, h, l, c = _ohlc(df)
    cl = _candle_length(h, l)
    med = _rolling_median(cl, n)
    is_long = cl >= med * threshold
    return pd.DataFrame(
        {"LongWhiteCandle": is_long & (c >= o),
         "LongBlackCandle": is_long & (o > c)},
        index=df.index,
    )


def add_long_candle_body(df: pd.DataFrame, n: int = 20,
                         threshold: float = 1.0) -> pd.DataFrame:
    """Long White / Black Candle Body — body length >= threshold × rolling median.

    Returns
    -------
    DataFrame with columns: LongWhiteCandleBody, LongBlackCandleBody (bool)
    """
    o, h, l, c = _ohlc(df)
    cbl = _candle_body_length(o, c)
    med = _rolling_median(cbl, n)
    is_long = cbl >= med * threshold
    return pd.DataFrame(
        {"LongWhiteCandleBody": is_long & (c >= o),
         "LongBlackCandleBody": is_long & (o > c)},
        index=df.index,
    )


def add_short_candle(df: pd.DataFrame, n: int = 20,
                     threshold: float = 1.0) -> pd.DataFrame:
    """Short White / Black Candle — candle length < threshold × rolling median.

    Returns
    -------
    DataFrame with columns: ShortWhiteCandle, ShortBlackCandle (bool)
    """
    o, h, l, c = _ohlc(df)
    cl = _candle_length(h, l)
    med = _rolling_median(cl, n)
    is_short = cl < med * threshold
    return pd.DataFrame(
        {"ShortWhiteCandle": is_short & (c >= o),
         "ShortBlackCandle": is_short & (o > c)},
        index=df.index,
    )


def add_short_candle_body(df: pd.DataFrame, n: int = 20,
                          threshold: float = 1.0) -> pd.DataFrame:
    """Short White / Black Candle Body — body length < threshold × rolling median.

    Returns
    -------
    DataFrame with columns: ShortWhiteCandleBody, ShortBlackCandleBody (bool)
    """
    o, h, l, c = _ohlc(df)
    cbl = _candle_body_length(o, c)
    med = _rolling_median(cbl, n)
    is_short = cbl < med * threshold
    return pd.DataFrame(
        {"ShortWhiteCandleBody": is_short & (c >= o),
         "ShortBlackCandleBody": is_short & (o > c)},
        index=df.index,
    )


# ─── 2-bar patterns (R-only) ─────────────────────────────────────────────────

def add_gap(df: pd.DataFrame, ignore_shadows: bool = False) -> pd.DataFrame:
    """Gap Up / Gap Down (Rising Window / Falling Window).

    Parameters
    ----------
    ignore_shadows : bool
        If True, only compare body ranges (open/close), ignoring wicks.
        Default False (compare full high/low range).

    Returns
    -------
    DataFrame with columns: GapUp, GapDown (bool)
    """
    o, h, l, c = _ohlc(df)
    if ignore_shadows:
        body_hi = pd.concat([o, c], axis=1).max(axis=1)
        body_lo = pd.concat([o, c], axis=1).min(axis=1)
        gap_up   = body_lo > body_hi.shift(1)
        gap_down = body_hi < body_lo.shift(1)
    else:
        gap_up   = l > h.shift(1)
        gap_down = h < l.shift(1)
    return pd.DataFrame(
        {"GapUp": gap_up, "GapDown": gap_down},
        index=df.index,
    )


def add_inside_day(df: pd.DataFrame) -> pd.DataFrame:
    """Inside Day — current high ≤ prior high AND current low ≥ prior low.

    Returns
    -------
    DataFrame with column: InsideDay (bool)
    """
    o, h, l, c = _ohlc(df)
    signal = (h <= h.shift(1)) & (l >= l.shift(1))
    return pd.DataFrame({"InsideDay": signal}, index=df.index)


def add_outside_day(df: pd.DataFrame) -> pd.DataFrame:
    """Outside Day — current high > prior high AND current low < prior low.

    Returns
    -------
    DataFrame with column: OutsideDay (bool)
    """
    o, h, l, c = _ohlc(df)
    signal = (h > h.shift(1)) & (l < l.shift(1))
    return pd.DataFrame({"OutsideDay": signal}, index=df.index)


def add_stomach(df: pd.DataFrame) -> pd.DataFrame:
    """Above / Below The Stomach — reversal patterns based on midpoint open.

    AboveTheStomach (bull): prior black, current white, current open ≥ prior body midpoint.
    BelowTheStomach (bear): prior white, current black, current open ≤ prior body midpoint.

    Returns
    -------
    DataFrame with columns: AboveTheStomach, BelowTheStomach (bool)
    """
    o, h, l, c = _ohlc(df)
    o1, c1 = o.shift(1), c.shift(1)
    mid1 = (o1 + c1) / 2

    above = (o1 > c1) & (c > o) & (o >= mid1)   # prior black, current white
    below = (c1 > o1) & (o > c) & (o <= mid1)   # prior white, current black

    return pd.DataFrame(
        {"AboveTheStomach": above, "BelowTheStomach": below},
        index=df.index,
    )


# ─── N-bar parameterised patterns ────────────────────────────────────────────

def add_n_higher_close(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """N Consecutive Higher Closes.

    Parameters
    ----------
    n : int
        Number of consecutive higher closes required. Default 3.

    Returns
    -------
    DataFrame with column: {n}HigherClose (bool)
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    o, h, l, c = _ohlc(df)
    signal = pd.Series(True, index=df.index)
    for k in range(1, n + 1):
        signal = signal & (c.shift(k - 1) > c.shift(k))
    col = f"{n}HigherClose"
    return pd.DataFrame({col: signal}, index=df.index)


def add_n_lower_close(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """N Consecutive Lower Closes.

    Parameters
    ----------
    n : int
        Number of consecutive lower closes required. Default 3.

    Returns
    -------
    DataFrame with column: {n}LowerClose (bool)
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    o, h, l, c = _ohlc(df)
    signal = pd.Series(True, index=df.index)
    for k in range(1, n + 1):
        signal = signal & (c.shift(k - 1) < c.shift(k))
    col = f"{n}LowerClose"
    return pd.DataFrame({col: signal}, index=df.index)


def _n_long_candles(df: pd.DataFrame, n: int, lookback: int,
                    threshold: float, white: bool) -> pd.Series:
    """Shared logic for N consecutive long white/black candles."""
    o, h, l, c = _ohlc(df)
    cl = _candle_length(h, l)
    med = _rolling_median(cl, lookback)
    is_long = cl >= med * threshold
    is_color = (c >= o) if white else (o > c)
    is_target = is_long & is_color
    # N consecutive: rolling sum equals N
    return is_target.rolling(n).sum() == n


def add_n_long_white_candles(df: pd.DataFrame, n: int = 3,
                              lookback: int = 20,
                              threshold: float = 1.0) -> pd.DataFrame:
    """N Consecutive Long White Candles.

    Returns
    -------
    DataFrame with column: {n}LongWhiteCandles (bool)
    """
    col = f"{n}LongWhiteCandles"
    sig = _n_long_candles(df, n, lookback, threshold, white=True)
    return pd.DataFrame({col: sig}, index=df.index)


def add_n_long_black_candles(df: pd.DataFrame, n: int = 3,
                              lookback: int = 20,
                              threshold: float = 1.0) -> pd.DataFrame:
    """N Consecutive Long Black Candles.

    Returns
    -------
    DataFrame with column: {n}LongBlackCandles (bool)
    """
    col = f"{n}LongBlackCandles"
    sig = _n_long_candles(df, n, lookback, threshold, white=False)
    return pd.DataFrame({col: sig}, index=df.index)


def _n_long_bodies(df: pd.DataFrame, n: int, lookback: int,
                   threshold: float, white: bool) -> pd.Series:
    """Shared logic for N consecutive long white/black candle bodies."""
    o, h, l, c = _ohlc(df)
    cbl = _candle_body_length(o, c)
    med = _rolling_median(cbl, lookback)
    is_long = cbl >= med * threshold
    is_color = (c >= o) if white else (o > c)
    is_target = is_long & is_color
    return is_target.rolling(n).sum() == n


def add_n_long_white_candle_bodies(df: pd.DataFrame, n: int = 3,
                                   lookback: int = 20,
                                   threshold: float = 1.0) -> pd.DataFrame:
    """N Consecutive Long White Candle Bodies.

    Returns
    -------
    DataFrame with column: {n}LongWhiteCandleBodies (bool)
    """
    col = f"{n}LongWhiteCandleBodies"
    sig = _n_long_bodies(df, n, lookback, threshold, white=True)
    return pd.DataFrame({col: sig}, index=df.index)


def add_n_long_black_candle_bodies(df: pd.DataFrame, n: int = 3,
                                   lookback: int = 20,
                                   threshold: float = 1.0) -> pd.DataFrame:
    """N Consecutive Long Black Candle Bodies.

    Returns
    -------
    DataFrame with column: {n}LongBlackCandleBodies (bool)
    """
    col = f"{n}LongBlackCandleBodies"
    sig = _n_long_bodies(df, n, lookback, threshold, white=False)
    return pd.DataFrame({col: sig}, index=df.index)


def add_n_blended(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Blended N-Bar OHLC — merge N candles into one synthetic candle.

    The blended candle:
    - Open  = Open of the candle N bars ago
    - High  = rolling maximum High over the past N bars
    - Low   = rolling minimum Low over the past N bars
    - Close = current Close

    Parameters
    ----------
    n : int
        Number of bars to blend. Default 3.

    Returns
    -------
    DataFrame with columns:
        {n}.Blended.Open, {n}.Blended.High, {n}.Blended.Low, {n}.Blended.Close
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    o, h, l, c = _ohlc(df)
    blended_open  = o.shift(n - 1)
    blended_high  = h.rolling(n).max()
    blended_low   = l.rolling(n).min()
    blended_close = c
    prefix = f"{n}"
    return pd.DataFrame(
        {
            f"{prefix}.Blended.Open":  blended_open,
            f"{prefix}.Blended.High":  blended_high,
            f"{prefix}.Blended.Low":   blended_low,
            f"{prefix}.Blended.Close": blended_close,
        },
        index=df.index,
    ).dropna()
