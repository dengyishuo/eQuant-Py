"""Utility functions for data preparation and validation."""

from typing import Optional, Union

import numpy as np
import pandas as pd


# Required OHLC column names (lowercase)
_REQUIRED_COLS = {"open", "high", "low", "close"}


def prepare_ohlc(
    data: pd.DataFrame,
    date_col: Optional[str] = None,
    volume_col: Optional[str] = "volume",
) -> pd.DataFrame:
    """Normalise input DataFrame for mplfinance consumption.

    Accepts various input formats:
    - DataFrame with index = DatetimeIndex, columns = [Open, High, Low, Close, Volume]
    - DataFrame with any index + a date column, columns = [Open | open, High | high, ...]
    - Columns are case-insensitive; will be renamed to lowercase.

    Parameters
    ----------
    data : pd.DataFrame
        OHLC(+V) data.
    date_col : str, optional
        If provided, use this column as the DatetimeIndex.
    volume_col : str, optional
        Name of the volume column (case-insensitive). Use None to skip volume.

    Returns
    -------
    pd.DataFrame
        Normalised DataFrame with DatetimeIndex, lowercase columns [open, high, low, close, volume(optional)].
    """
    df = data.copy()

    # Handle date column
    if date_col is not None:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        else:
            # date_col might not exist as a column name — could be index name
            pass

    # If index is not datetime, try to convert
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError(
                "Cannot convert index to DatetimeIndex. "
                "Provide a date_col parameter with the date column name."
            )

    # Handle MultiIndex columns (e.g., from yfinance auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten: take the first level (e.g., ('Close', 'AAPL') -> 'Close')
        df.columns = [c[0] for c in df.columns]

    # Normalize column names to lowercase
    col_map = {}
    for col in df.columns:
        col_str = str(col)
        lower = col_str.lower()
        if lower in ("open", "high", "low", "close", "volume"):
            col_map[col] = lower
    df = df.rename(columns=col_map)

    # Validate required columns
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"DataFrame must contain Open, High, Low, Close (case-insensitive)."
        )

    # Select relevant columns
    cols = ["open", "high", "low", "close"]
    if volume_col is not None:
        vol_lower = volume_col.lower()
        if vol_lower in df.columns:
            cols.append(vol_lower)

    return df[cols].sort_index()


def make_addplot(
    data: Union[pd.Series, pd.DataFrame, list],
    panel: int = 0,
    color: Optional[str] = None,
    secondary_y: bool = False,
    width: Optional[float] = None,
    alpha: Optional[float] = None,
    linestyle: Optional[str] = None,
    marker: Optional[str] = None,
    markersize: Optional[float] = None,
    type: str = "line",
    **kwargs,
) -> dict:
    """Build a dict suitable for mpf.make_addplot().

    Convenience wrapper that returns a dict rather than the mpf object,
    allowing deferred style application.
    """
    ap = {"data": data, "panel": panel, "secondary_y": secondary_y}
    if color is not None:
        ap["color"] = color
    if width is not None:
        ap["width"] = width
    if alpha is not None:
        ap["alpha"] = alpha
    if linestyle is not None:
        ap["linestyle"] = linestyle
    if marker is not None:
        ap["marker"] = marker
    if markersize is not None:
        ap["markersize"] = markersize
    if type != "line":
        ap["type"] = type
    ap.update(kwargs)
    return ap


def make_addplots(
    *addplot_dicts: dict,
) -> list:
    """Convert addplot dicts to mplfinance addplot objects.

    Parameters
    ----------
    *addplot_dicts : dict
        Dicts built by make_addplot().

    Returns
    -------
    list
        List of mplfinance addplot objects.
    """
    import mplfinance as mpf

    result = []
    for ap in addplot_dicts:
        result.append(mpf.make_addplot(**ap))
    return result
