"""Data acquisition — eFactorCraft get_data equivalent.

Downloads OHLCV data from Yahoo Finance, Tushare, AKShare, or baostock and
returns a standardized long-format panel DataFrame.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

_SOURCES = ("yahoo", "tushare", "akshare", "baostock")


def get_data(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    source: str = "yahoo",
    progress: bool = True,
) -> pd.DataFrame:
    """Download OHLCV data for multiple assets and return long-format panel.

    Parameters
    ----------
    df : DataFrame
        Must have ``code`` and ``name`` columns. Each row is one asset.
        For ``source="yahoo"``, ``code`` is a Yahoo ticker (e.g. ``"AAPL"``).
        For ``"tushare"``/``"akshare"``/``"baostock"``, ``code`` is a bare
        6-digit A-share code (e.g. ``"600000"``); each provider converts it
        to its own convention internally.
    start_date : str
        Start date in ``YYYY-MM-DD`` format.
    end_date : str
        End date in ``YYYY-MM-DD`` format.
    source : str
        One of ``"yahoo"`` (default), ``"tushare"``, ``"akshare"``, ``"baostock"``.
        Tushare requires a token set via ``efactorcraft.providers.set_token()``
        beforehand.
    progress : bool
        Show download progress.

    Returns
    -------
    DataFrame
        Long-format panel with columns:
        ``date``, ``code``, ``name``, ``open``, ``high``, ``low``,
        ``close``, ``adjusted``, ``volume``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a DataFrame with 'code' and 'name' columns")

    required = {"code", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"'df' missing columns: {missing}")

    if len(df) == 0:
        raise ValueError("'df' must contain at least one asset")

    if source not in _SOURCES:
        raise ValueError(f"'source' must be one of {_SOURCES}, got {source!r}")

    if source != "yahoo":
        return _get_data_cn(df, start_date, end_date, source=source, progress=progress)

    import yfinance as yf

    frames: list[pd.DataFrame] = []

    for _, row in df.iterrows():
        code = row["code"]
        name = row["name"]

        if progress:
            print(f"Downloading: {code} | {name}")

        try:
            ticker = yf.Ticker(code)
            hist = ticker.history(start=start_date, end=end_date)

            if hist.empty:
                print(f"  Warning: No data for {code}")
                continue

            # Normalize columns to lowercase standard names
            hist = hist.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # Handle Adjusted close (may be named differently)
            if "Adj Close" in hist.columns:
                hist = hist.rename(columns={"Adj Close": "adjusted"})
            elif "adjusted" not in hist.columns:
                hist["adjusted"] = hist["close"]

            # Forward-fill price gaps, interpolate volume
            for c in ["open", "high", "low", "close", "adjusted"]:
                if c in hist.columns:
                    hist[c] = hist[c].ffill()

            if "volume" in hist.columns:
                hist["volume"] = hist["volume"].interpolate().fillna(0)

            # Reset index to get date as column
            hist = hist.reset_index()
            hist = hist.rename(columns={"Date": "date", "Datetime": "date"})
            hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)

            hist["code"] = code
            hist["name"] = name

            # Keep only standard columns
            keep = ["date", "code", "name", "open", "high", "low", "close", "adjusted", "volume"]
            hist = hist[[c for c in keep if c in hist.columns]]

            frames.append(hist)

        except Exception as e:
            print(f"  Warning: Download failed for {code}: {e}")
            continue

    if not frames:
        raise RuntimeError("No data downloaded for any asset in the stock list")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["date", "code"]).reset_index(drop=True)

    print(f"Download complete. Total rows: {len(result)}")
    return result


def _get_data_cn(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    source: str,
    progress: bool = True,
) -> pd.DataFrame:
    """Shared per-asset download loop for the three A-share providers.

    Each provider's ``get_daily(code, start_date, end_date)`` accepts
    ``YYYY-MM-DD`` and internally converts to whatever format its own API
    needs (tushare/akshare want ``YYYYMMDD``, baostock wants the dashed
    form) — this loop passes dates through unchanged and only handles
    looping over the universe, attaching ``name``, and stacking results.
    """
    from efactorcraft.providers import akshare, baostock, tushare
    from efactorcraft.providers._codes import bare

    provider = {"tushare": tushare, "akshare": akshare, "baostock": baostock}[source]

    frames: list[pd.DataFrame] = []

    for _, row in df.iterrows():
        code = bare(str(row["code"]))
        name = row["name"]

        if progress:
            print(f"Downloading: {code} | {name} (source={source})")

        try:
            hist = provider.get_daily(code, start_date, end_date)
        except Exception as e:
            print(f"  Warning: Download failed for {code}: {e}")
            continue

        if hist.empty:
            print(f"  Warning: No data for {code}")
            continue

        hist = hist.copy()
        hist["name"] = name
        for c in ["open", "high", "low", "close", "adjusted"]:
            hist[c] = hist[c].ffill()
        hist["volume"] = hist["volume"].interpolate().fillna(0)

        keep = ["date", "code", "name", "open", "high", "low", "close", "adjusted", "volume"]
        frames.append(hist[keep])

    if not frames:
        raise RuntimeError("No data downloaded for any asset in the stock list")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["date", "code"]).reset_index(drop=True)

    print(f"Download complete. Total rows: {len(result)}")
    return result
