"""Tushare Pro data source adapter — eTushare equivalent.

Provides idiomatic Python wrappers around the Tushare Pro REST API
for Chinese financial market data (A-shares, indices, industry sectors,
trading calendar, financial indicators).

Token Management
----------------
Set your Tushare token via environment variable ``TUSHARE_TOKEN``
or by calling ``set_token("your_token_here")``.

Usage::

    from efactorcraft.providers import tushare

    tushare.set_token("your_api_token")

    df = tushare.get_index_daily(ts_code="000001.SH", start_date="20230101")
    df = tushare.get_info(market="main", list_status="L")
    df = tushare.get_trading_date(exchange="SSE")
"""

from __future__ import annotations

import os
import time
from typing import Optional, Sequence

import pandas as pd

from efactorcraft.providers._codes import to_tushare

# ── Module-level token ───────────────────────────────────────────────────
_token: Optional[str] = os.environ.get("TUSHARE_TOKEN")

# Rate limiting: Tushare Pro allows ~200 calls/minute for free users
_last_call_time: float = 0.0
_MIN_INTERVAL: float = 0.35  # seconds between calls


def set_token(token: str) -> None:
    """Set the Tushare Pro API token."""
    global _token
    _token = token


def get_token() -> Optional[str]:
    """Get the current Tushare Pro API token."""
    return _token


# ══════════════════════════════════════════════════════════════════════════
# Internal HTTP client
# ══════════════════════════════════════════════════════════════════════════

import requests

_TUSHARE_URL = "https://api.tushare.pro"


def _call_api(api_name: str, params: dict, fields: Sequence[str]) -> pd.DataFrame:
    """Call the Tushare Pro API and return results as a DataFrame.

    Handles token injection, rate limiting, pagination, and error checking.
    """
    global _last_call_time

    token = _token
    if not token:
        raise ValueError(
            "Tushare token not set. Call tushare.set_token('your_token') "
            "or set the TUSHARE_TOKEN environment variable."
        )

    # Rate limiting
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)

    # Build request
    payload = {
        "api_name": api_name,
        "token": token,
        "params": {k: v for k, v in params.items() if v is not None},
        "fields": ",".join(fields),
    }

    resp = requests.post(_TUSHARE_URL, json=payload, timeout=30)
    _last_call_time = time.time()

    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Tushare API error [{api_name}]: {data.get('msg', 'Unknown error')}")

    items = data.get("data", {}).get("items", [])
    if not items:
        return pd.DataFrame(columns=list(fields))

    df = pd.DataFrame(items, columns=list(fields))

    # Replace empty strings with NaN
    df = df.replace("", pd.NA)

    return df


# ══════════════════════════════════════════════════════════════════════════
# Public API functions
# ══════════════════════════════════════════════════════════════════════════


def get_index_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Daily OHLCV for Chinese A-share market indices.

    Wraps Tushare ``index_daily`` endpoint.

    Parameters
    ----------
    ts_code : str, optional
        Index code (e.g., ``"000001.SH"`` for SSE Composite).
    trade_date : str, optional
        Single trading date ``YYYYMMDD``.
    start_date : str, optional
        Start date ``YYYYMMDD``.
    end_date : str, optional
        End date ``YYYYMMDD``.
    fields : sequence of str, optional
        Fields to fetch. Defaults to OHLCV + change fields.

    Returns
    -------
    DataFrame
    """
    if fields is None:
        fields = [
            "ts_code", "trade_date", "close", "open", "high", "low",
            "pre_close", "change", "pct_chg", "vol", "amount",
        ]

    # Ensure ts_code is in fields
    if "ts_code" not in fields:
        fields = ["ts_code"] + list(fields)

    params = {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "start_date": start_date,
        "end_date": end_date,
    }

    df = _call_api("index_daily", params, fields)

    # Client-side filter
    if ts_code and len(df) > 0 and "ts_code" in df.columns:
        df = df[df["ts_code"] == ts_code]

    return df


def get_sw_daily(
    ts_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Daily quotes for Shenwan industry indices.

    Wraps Tushare ``sw_daily`` endpoint.

    Parameters
    ----------
    ts_code : str, optional
        Shenwan index code (e.g., ``"801010.SI"``).
    fields : sequence of str, optional
        Includes valuation fields (pe, pb, float_mv, total_mv) by default.
    """
    if fields is None:
        fields = [
            "ts_code", "trade_date", "close", "open", "high", "low",
            "pre_close", "change", "pct_chg", "vol", "amount",
            "pe", "pb", "float_mv", "total_mv",
        ]

    if "ts_code" not in fields:
        fields = ["ts_code"] + list(fields)

    params = {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "start_date": start_date,
        "end_date": end_date,
    }

    return _call_api("sw_daily", params, fields)


def get_trading_date(
    exchange: str = "",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    is_open: Optional[str] = None,
) -> pd.DataFrame:
    """Chinese market trading calendar.

    Wraps Tushare ``trade_cal`` endpoint.

    Parameters
    ----------
    exchange : str
        Exchange code: ``"SSE"``, ``"SZSE"``, ``"BSE"``, ``"CFFEX"``,
        ``"SHFE"``, ``"CZCE"``, ``"DCE"``, ``"INE"``.
    start_date, end_date : str, optional
        Date range ``YYYYMMDD``.
    is_open : str, optional
        ``"0"`` = holiday, ``"1"`` = trading day.
    """
    params = {
        "exchange": exchange if exchange else None,
        "start_date": start_date,
        "end_date": end_date,
        "is_open": is_open,
    }
    fields = ["exchange", "cal_date", "is_open", "pretrade_date"]

    return _call_api("trade_cal", params, fields)


def get_info(
    ts_code: str = "",
    market: str = "",
    list_status: str = "L",
    exchange: str = "",
    is_hs: str = "",
    fields: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Stock basic information — the master table of Chinese stocks.

    Wraps Tushare ``stock_basic`` endpoint.

    Parameters
    ----------
    ts_code : str, optional
        Stock code with ``.SH`` / ``.SZ`` suffix.
    market : str, optional
        Market board. One of: ``"main"``, ``"gem"``, ``"star"``, ``"cdr"``, ``"bse"``.
        These English keywords are translated to Chinese market names.
    list_status : str
        ``"L"`` = listed, ``"D"`` = delisted, ``"P"`` = paused.
    exchange : str, optional
        Exchange filter: ``"SSE"``, ``"SZSE"``, ``"BSE"``.
    is_hs : str, optional
        Stock Connect eligibility: ``"N"``, ``"H"``, ``"S"``.
    """
    # Market mapping (English → Chinese, matching R implementation)
    market_map = {
        "main": "主板",
        "gem": "创业板",
        "star": "科创板",
        "cdr": "CDR",
        "bse": "北交所",
    }

    # Validate ts_code format
    if ts_code and not (".SH" in ts_code.upper() or ".SZ" in ts_code.upper()):
        raise ValueError("Invalid stock code format, must include .SH or .SZ suffix")

    # Validate and translate market
    if market and market not in market_map:
        valid = ", ".join(market_map.keys())
        raise ValueError(f"market must be one of: {valid}")
    market_cn = market_map.get(market) if market else None

    # Validate is_hs
    if is_hs:
        is_hs = is_hs.upper()
        if is_hs not in ("N", "H", "S"):
            raise ValueError("is_hs must be one of: N, H, S")

    if fields is None:
        fields = ["ts_code", "symbol", "name", "area", "industry", "market", "list_date"]

    if "ts_code" not in fields:
        fields = ["ts_code"] + list(fields)

    params = {
        "ts_code": ts_code if ts_code else None,
        "market": market_cn,
        "list_status": list_status,
        "exchange": exchange if exchange else None,
        "is_hs": is_hs if is_hs else None,
    }

    df = _call_api("stock_basic", params, fields)

    if ts_code and len(df) > 0 and "ts_code" in df.columns:
        df = df[df["ts_code"] == ts_code]

    return df


def get_capital(
    ts_code: str = "",
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Pre-market data — total shares, float shares, limit prices.

    Wraps Tushare ``stk_premarket`` endpoint.
    """
    if fields is None:
        fields = [
            "ts_code", "trade_date", "total_share", "float_share",
            "pre_close", "up_limit", "down_limit",
        ]

    if "ts_code" not in fields:
        fields = ["ts_code"] + list(fields)

    params = {
        "ts_code": ts_code if ts_code else None,
        "trade_date": trade_date,
        "start_date": start_date,
        "end_date": end_date,
    }

    df = _call_api("stk_premarket", params, fields)

    if ts_code and len(df) > 0 and "ts_code" in df.columns:
        df = df[df["ts_code"] == ts_code]

    return df


def get_financial_indicators(
    ts_code: str = "600519.SH",
    ann_date: Optional[str] = None,
    fann_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    report_type: Optional[str] = None,
    comp_type: Optional[str] = None,
    fields: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Financial statement indicators (ROE, EPS, debt ratios, etc.).

    Wraps Tushare ``fina_indicator`` endpoint.

    Parameters
    ----------
    ts_code : str
        Stock code. Default ``"600519.SH"`` (Moutai).
    period : str, optional
        Reporting period: ``"20231231"`` for annual.
    report_type : str, optional
        ``"1"`` = consolidated (default), ``"2"`` = parent company.
    comp_type : str, optional
        ``"1"`` = YoY, ``"2"`` = QoQ.
    """
    if ts_code and not (".SH" in ts_code.upper() or ".SZ" in ts_code.upper()):
        raise ValueError("Invalid stock code format, must include .SH or .SZ suffix")

    if fields is None:
        fields = [
            "ts_code", "ann_date", "end_date", "period",
            "eps", "roe", "debt_to_asset", "operate_rev", "profit_dedt",
        ]

    if "ts_code" not in fields:
        fields = ["ts_code"] + list(fields)

    params = {
        "ts_code": ts_code,
        "ann_date": ann_date,
        "fann_date": fann_date,
        "start_date": start_date,
        "end_date": end_date,
        "period": period,
        "report_type": report_type,
        "comp_type": comp_type,
    }

    df = _call_api("fina_indicator", params, fields)

    if ts_code and len(df) > 0 and "ts_code" in df.columns:
        df = df[df["ts_code"] == ts_code]

    return df


def get_daily(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Daily OHLCV + adjusted close for a single A-share stock.

    Wraps the Tushare ``daily`` and ``adj_factor`` endpoints. Tushare's
    ``daily`` returns raw (unadjusted) prices; the forward-adjusted
    ``adjusted`` column is derived by scaling ``close`` with ``adj_factor``.

    Parameters
    ----------
    code : str
        Bare 6-digit A-share code (e.g. ``"600000"``) or a full
        Tushare ``ts_code`` (e.g. ``"600000.SH"``).
    start_date, end_date : str
        ``YYYYMMDD`` or ``YYYY-MM-DD`` (dashes are stripped internally).

    Returns
    -------
    DataFrame
        Columns: ``date, code, open, high, low, close, adjusted, volume``.
    """
    ts_code = code if "." in code else to_tushare(code)
    start_date = start_date.replace("-", "")
    end_date = end_date.replace("-", "")

    fields = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
    params = {"ts_code": ts_code, "start_date": start_date, "end_date": end_date}
    df = _call_api("daily", params, fields)

    if df.empty:
        return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "adjusted", "volume"])

    adj = _call_api(
        "adj_factor", params, ["ts_code", "trade_date", "adj_factor"]
    )
    df = df.merge(adj[["trade_date", "adj_factor"]], on="trade_date", how="left")

    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    # adj_factor accumulates from listing date (base = 1.0); forward-fill any
    # gaps so every trading day has a usable factor.
    df["adj_factor"] = df["adj_factor"].astype(float).ffill().fillna(1.0)
    df["adjusted"] = df["close"] * df["adj_factor"]

    df = df.rename(columns={"ts_code": "code", "trade_date": "date", "vol": "volume"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df[["date", "code", "open", "high", "low", "close", "adjusted", "volume"]]
    return df.sort_values("date").reset_index(drop=True)
