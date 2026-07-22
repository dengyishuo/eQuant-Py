"""AKShare data source adapter — daily OHLCV for A-share stocks.

Usage::

    from efactorcraft.providers import akshare

    df = akshare.get_daily("600000", "20240101", "20240115")
"""

from __future__ import annotations

import pandas as pd

from efactorcraft.providers._codes import bare

_COLUMN_MAP = {
    "日期": "date",
    "股票代码": "code",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
}


def get_daily(code: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    """Daily OHLCV for a single A-share stock via ``akshare.stock_zh_a_hist``.

    Parameters
    ----------
    code : str
        Bare 6-digit A-share code, e.g. ``"600000"``.
    start_date, end_date : str
        ``YYYYMMDD`` or ``YYYY-MM-DD`` (akshare accepts both digit forms;
        dashes are stripped here for consistency).
    adjust : str
        ``"qfq"`` (forward-adjusted, default), ``"hfq"``, or ``""`` (raw).

    Returns
    -------
    DataFrame
        Columns: ``date, code, open, high, low, close, adjusted, volume``.
        Under ``qfq``/``hfq`` adjustment the whole OHLC series is already
        adjusted, so ``adjusted`` is set equal to ``close``.
    """
    import akshare as ak

    code = bare(code)
    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust=adjust,
    )

    if df.empty:
        return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "adjusted", "volume"])

    df = df.rename(columns=_COLUMN_MAP)[list(_COLUMN_MAP.values())]
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = code
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["adjusted"] = df["close"]

    df = df[["date", "code", "open", "high", "low", "close", "adjusted", "volume"]]
    return df.sort_values("date").reset_index(drop=True)
