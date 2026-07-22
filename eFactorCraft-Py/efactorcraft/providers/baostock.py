"""baostock data source adapter — daily OHLCV for A-share stocks.

Usage::

    from efactorcraft.providers import baostock

    df = baostock.get_daily("600000", "2024-01-01", "2024-01-15")
"""

from __future__ import annotations

import pandas as pd

from efactorcraft.providers._codes import bare, to_baostock

_FIELDS = ["date", "code", "open", "high", "low", "close", "volume"]


def get_daily(code: str, start_date: str, end_date: str, adjustflag: str = "2") -> pd.DataFrame:
    """Daily OHLCV for a single A-share stock via ``baostock.query_history_k_data_plus``.

    Parameters
    ----------
    code : str
        Bare 6-digit A-share code, e.g. ``"600000"``.
    start_date, end_date : str
        ``YYYY-MM-DD``.
    adjustflag : str
        ``"1"`` = backward-adjusted (hfq), ``"2"`` = forward-adjusted (qfq,
        default), ``"3"`` = raw/unadjusted.

    Returns
    -------
    DataFrame
        Columns: ``date, code, open, high, low, close, adjusted, volume``.
        Under qfq/hfq the whole OHLC series is already adjusted, so
        ``adjusted`` is set equal to ``close``.
    """
    import baostock as bs

    bs_code = to_baostock(code)

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            ",".join(_FIELDS),
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag,
        )
        if rs.error_code != "0":
            raise RuntimeError(f"baostock query failed: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()

    if not rows:
        return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "adjusted", "volume"])

    df = pd.DataFrame(rows, columns=_FIELDS)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = bare(code)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["adjusted"] = df["close"]

    df = df[["date", "code", "open", "high", "low", "close", "adjusted", "volume"]]
    return df.sort_values("date").reset_index(drop=True)
