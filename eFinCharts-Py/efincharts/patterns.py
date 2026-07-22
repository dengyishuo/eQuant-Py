"""CSP (Candlestick Pattern) marker overlays for price charts.

Adds visual markers (arrows, dots, labels) for detected candlestick patterns
on the candlestick chart.  Integrates with eCandleSticks-Py for pattern detection,
but accepts any boolean-indexed DataFrame of signals.

Usage
-----
>>> from efincharts import candlestick
>>> chart = candlestick(df, volume=True)

>>> # With eCandleSticks-Py:
>>> from eCandleSticks import detect_hammer, detect_engulfing
>>> hammer = detect_hammer(df)
>>> engulfing = detect_engulfing(df)
>>> chart.add_pattern(hammer, name="Hammer", bull=True)
>>> chart.add_pattern(engulfing, name="Engulfing", bull=None)
>>> chart.show()
"""

from typing import Optional

import numpy as np
import pandas as pd

from .theme import COLOR_PATTERN_BULL, COLOR_PATTERN_BEAR


def _prepare_pattern_markers(
    signals: pd.DataFrame,
    data: pd.DataFrame,
    name: str = "pattern",
    bull: Optional[bool] = True,
) -> Optional[list[dict]]:
    """Prepare mplfinance addplot dicts for CSP pattern markers.

    Parameters
    ----------
    signals : pd.DataFrame
        Signal DataFrame from eCandleSticks-Py or manual detection.
        Must have at least one boolean column, or a column named 'signal'.
        Index must be DatetimeIndex or date-like.
    data : pd.DataFrame
        The OHLC data DataFrame (to align index and get Low/High for marker placement).
    name : str
        Pattern display name (used in legend).
    bull : bool or None
        - True: bullish pattern (blue upward arrow below candle)
        - False: bearish pattern (red downward arrow above candle)
        - None: auto-detect from pattern name or use a neutral marker

    Returns
    -------
    list of dict, or None if no signals found.
        Each dict is suitable for mpf.make_addplot(**dict).
    """
    # Find the signal column
    signal_col = None
    for col in signals.columns:
        if col.lower() in ("signal", "pattern", "is_pattern"):
            signal_col = col
            break
    if signal_col is None:
        # Use the first boolean column
        for col in signals.columns:
            if signals[col].dtype == bool:
                signal_col = col
                break
    if signal_col is None:
        # Use the first column
        signal_col = signals.columns[0]

    # Filter to True signals
    if signals.index.name is None or signals.index.name.lower() not in ("date", "time"):
        sig_idx = signals.index
    else:
        sig_idx = signals.index

    true_signals = signals.loc[signals[signal_col] == True]  # noqa: E712
    if len(true_signals) == 0:
        return None

    # Align with price data
    common_idx = true_signals.index.intersection(data.index)
    if len(common_idx) == 0:
        return None

    true_signals = true_signals.loc[common_idx]

    # Determine color and marker placement
    if bull is None:
        # Auto-detect: check if name contains bullish/bearish keywords
        name_lower = name.lower()
        if any(kw in name_lower for kw in ("bull", "hammer", "soldier", "engulf", "piercing", "morning", "white")):
            bull = True
        elif any(kw in name_lower for kw in ("bear", "crow", "dark", "shooting", "evening", "black", "hanging")):
            bull = False
        else:
            bull = True  # default to bullish color

    color = COLOR_PATTERN_BULL if bull else COLOR_PATTERN_BEAR
    marker = "^" if bull else "v"

    # Place marker below the low (bullish) or above the high (bearish)
    if bull:
        y_offset = data.loc[common_idx, "low"] * 0.99
    else:
        y_offset = data.loc[common_idx, "high"] * 1.01

    # Build marker series (NaN everywhere except at signal points)
    marker_series = pd.Series(np.nan, index=data.index)
    marker_series.loc[common_idx] = y_offset.loc[common_idx]

    return [
        dict(
            data=marker_series,
            type="scatter",
            marker=marker,
            markersize=80,
            color=color,
            alpha=0.85,
            label=name,
        )
    ]
