"""Internal helpers: OHLC normalisation and rolling average."""

import pandas as pd
import numpy as np


def _ohlc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (Open, High, Low, Close) series from a DataFrame."""
    cols = {c.lower(): c for c in df.columns}
    missing = [k for k in ("open", "high", "low", "close") if k not in cols]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    o = df[cols["open"]].astype(float)
    h = df[cols["high"]].astype(float)
    l = df[cols["low"]].astype(float)
    c = df[cols["close"]].astype(float)
    return o, h, l, c


def _rolling_mean(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()
