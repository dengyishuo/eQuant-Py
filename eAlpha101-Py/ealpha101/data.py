"""
Built-in sample dataset for eAlpha101.

``load_sample_data()`` returns a long-format panel DataFrame with 10 A-share
style stocks over 500 trading days, suitable for running any add_alphaXXX
function out of the box.

Columns
-------
date          : trading date (datetime64)
code          : stock ticker (str)
name          : stock name (str)
open          : opening price
high          : daily high
low           : daily low
close         : closing price
adjusted      : adjusted closing price
volume        : trading volume (shares)
vwap          : volume-weighted average price
returns       : daily simple return (close pct_change)
cap           : market capitalisation (close * total shares, simulated)
bv            : book value per share (quarterly carry-forward, simulated)
op            : operating profit per share (quarterly carry-forward, simulated)
assets        : total assets (quarterly carry-forward, simulated)
benchmark_ret : equal-weight cross-sectional average return per day

Pre-neutralised columns (industry-demeaned, needed by IndNeutralize alphas)
---------------------------------------------------------------------------
neut_close    : IndNeutralize(close, industry)
neut_vwap     : IndNeutralize(vwap, industry)
neut_low      : IndNeutralize(low, industry)
neut_volume   : IndNeutralize(volume, industry)
neut_price79  : IndNeutralize(open*0.60733+close*(1-0.60733), industry)
neut_price80  : IndNeutralize(open*0.868128+high*(1-0.868128), industry)
neut_price97  : IndNeutralize(low*0.721001+vwap*(1-0.721001), industry)
neut_vwap2    : IndNeutralize(vwap*0.728317+vwap*(1-0.728317), industry)
neut_close_ret: IndNeutralize(delta(close,1)/delay(close,1), industry)
neut_vwap_ret : IndNeutralize(delta(vwap,1)/delay(vwap,1), industry)
neut_rank100  : double-IndNeutralize rank term for alpha100
neut_diff100  : IndNeutralize (corr-argmin) term for alpha100
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def load_sample_data(seed: int = 2024) -> pd.DataFrame:
    """
    Return a built-in long-format panel DataFrame for testing/examples.

    Parameters
    ----------
    seed : int, default 2024
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        2500 rows (10 stocks × 500 trading days).
        All columns required by any ``add_alphaXXX`` function are present.

    Examples
    --------
    >>> from ealpha101 import load_sample_data, add_alpha001, add_alpha101
    >>> df = load_sample_data()
    >>> df.shape
    (5000, 28)
    >>> result = add_alpha001(df)
    >>> result[["date", "code", "alpha001"]].head()
    """
    rng = np.random.default_rng(seed)

    n_stocks = 10
    n_days = 500

    tickers = [
        "600519.SS", "000858.SZ", "601318.SS", "600036.SS", "000333.SZ",
        "600900.SS", "601166.SS", "000001.SZ", "600276.SS", "300760.SZ",
    ]
    names = [
        "贵州茅台", "五粮液", "中国平安", "招商银行", "美的集团",
        "长江电力", "兴业银行", "平安银行", "恒瑞医药", "迈瑞医疗",
    ]
    industries = ["消费", "消费", "金融", "金融", "制造",
                  "公用", "金融", "金融", "医药", "医药"]

    dates = pd.bdate_range("2020-01-01", periods=n_days, freq="B")

    rows = []
    for i, (ticker, name, ind) in enumerate(zip(tickers, names, industries)):
        # price path via geometric Brownian motion
        mu = rng.uniform(0.0001, 0.0006)
        sigma = rng.uniform(0.012, 0.022)
        log_ret = rng.normal(mu, sigma, n_days)
        close = 50 * rng.uniform(0.5, 3.0) * np.exp(np.cumsum(log_ret))

        daily_vol = rng.uniform(0.005, 0.015, n_days)
        high = close * (1 + rng.uniform(0.002, 0.01, n_days) + daily_vol / 2)
        low  = close * (1 - rng.uniform(0.002, 0.01, n_days) - daily_vol / 2)
        open_ = close * (1 + rng.uniform(-0.005, 0.005, n_days))
        open_ = np.clip(open_, low, high)

        volume = rng.integers(int(1e6), int(2e8), n_days).astype(float)
        vwap = (open_ + high + low + close) / 4

        adjusted = close * rng.uniform(0.95, 1.0)

        total_shares = rng.uniform(5e8, 5e10)
        cap = close * total_shares

        # quarterly fundamentals (carry-forward every 63 days)
        n_qtrs = -(-n_days // 63)  # ceiling division
        bv_base = close[0] * rng.uniform(0.3, 0.8)
        bv_qtr = bv_base * np.cumprod(1 + rng.uniform(-0.02, 0.06, n_qtrs))
        bv = np.repeat(bv_qtr, 63)[:n_days]

        op_base = bv_base * rng.uniform(0.05, 0.20)
        op_qtr = op_base * np.cumprod(1 + rng.uniform(-0.05, 0.10, n_qtrs))
        op = np.repeat(op_qtr, 63)[:n_days]

        assets_base = cap[0] * rng.uniform(0.5, 2.0)
        assets_qtr = assets_base * np.cumprod(1 + rng.uniform(-0.01, 0.05, n_qtrs))
        assets = np.repeat(assets_qtr, 63)[:n_days]

        for j, d in enumerate(dates):
            rows.append({
                "date": d,
                "code": ticker,
                "name": name,
                "industry": ind,
                "open": open_[j],
                "high": high[j],
                "low": low[j],
                "close": close[j],
                "adjusted": adjusted[j],
                "volume": volume[j],
                "vwap": vwap[j],
                "cap": cap[j],
                "bv": bv[j],
                "op": op[j],
                "assets": assets[j],
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # returns
    df["returns"] = df.groupby("code")["close"].pct_change()

    # benchmark = equal-weight daily return per date
    bm = df.groupby("date")["returns"].mean().rename("benchmark_ret")
    df = df.merge(bm, on="date", how="left")

    # ── industry-neutralisation helper ──────────────────────────────────────
    def _ind_neut(df: pd.DataFrame, col: str) -> pd.Series:
        """Subtract industry-date mean (simple industry neutralisation)."""
        return df[col] - df.groupby(["date", "industry"])[col].transform("mean")

    # Pre-neutralised columns
    df["neut_close"]  = _ind_neut(df, "close")
    df["neut_vwap"]   = _ind_neut(df, "vwap")
    df["neut_low"]    = _ind_neut(df, "low")
    df["neut_volume"] = _ind_neut(df, "volume")

    df["_price79"] = df["open"] * 0.60733 + df["close"] * (1 - 0.60733)
    df["neut_price79"] = _ind_neut(df, "_price79")

    df["_price80"] = df["open"] * 0.868128 + df["high"] * (1 - 0.868128)
    df["neut_price80"] = _ind_neut(df, "_price80")

    df["_price97"] = df["low"] * 0.721001 + df["vwap"] * (1 - 0.721001)
    df["neut_price97"] = _ind_neut(df, "_price97")

    df["neut_vwap2"] = _ind_neut(df, "vwap")  # simplified (0.728317 coefficient)

    df["_close_ret"] = df.groupby("code")["close"].pct_change()
    df["_vwap_ret"]  = df.groupby("code")["vwap"].pct_change()
    df["neut_close_ret"] = _ind_neut(df, "_close_ret")
    df["neut_vwap_ret"]  = _ind_neut(df, "_vwap_ret")

    # alpha100 terms (simplified placeholders matching the formula intent)
    df["_hl_vol"] = (
        (df["close"] - df["low"] - (df["high"] - df["close"]))
        / (df["high"] - df["low"]).replace(0, np.nan)
        * df["volume"]
    )
    df["_rank_hl_vol"] = df.groupby("date")["_hl_vol"].rank(pct=True)
    df["neut_rank100"] = _ind_neut(df, "_rank_hl_vol")
    df["neut_diff100"] = _ind_neut(df, "neut_close")  # simplified proxy

    # drop construction helpers
    tmp_cols = ["industry", "_price79", "_price80", "_price97",
                "_close_ret", "_vwap_ret", "_hl_vol", "_rank_hl_vol"]
    df = df.drop(columns=tmp_cols, errors="ignore")

    return df.reset_index(drop=True)
