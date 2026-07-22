"""Alpha #021 – #040"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ._base import _validate, _sort, _finish
from .utils import (
    cs_rank, scale_alpha, ts_rank, ts_mean, ts_stddev, ts_argmax, ts_argmin,
    ts_sum, ts_max, ts_min, ts_product, delta, delay, correlation,
    covariance, decay_linear, signedpower, adv,
)


def add_alpha021(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #021:
        if mean(close,8) + stddev(close,8) < mean(close,2): 1
        elif mean(close,2) < mean(close,8) - stddev(close,8): -1
        else: if volume/adv20 >= 1: 1, else: -1

    Required columns: date, code, name, ``close_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", close_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        m8 = ts_mean(g[close_col], 8)
        s8 = ts_stddev(g[close_col], 8)
        m2 = ts_mean(g[close_col], 2)
        adv20 = adv(g[volume_col], 20)
        vol_ratio = g[volume_col] / adv20.replace(0, np.nan)
        cond1 = m8 + s8 < m2
        cond2 = m2 < m8 - s8
        else_val = np.where(vol_ratio >= 1, 1.0, -1.0)
        return pd.Series(
            np.where(cond1, 1.0, np.where(cond2, -1.0, else_val)),
            index=g.index,
        )

    df["alpha021"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha021", append, [])


def add_alpha022(
    df: pd.DataFrame,
    high_col: str = "high",
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #022: -1 * delta(correlation(high,volume,5),5) * rank(stddev(close,20))

    Required columns: date, code, name, ``high_col``, ``close_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", high_col, close_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        corr = correlation(g[high_col], g[volume_col], 5)
        d_corr = delta(corr, 5)
        std20 = ts_stddev(g[close_col], 20)
        return pd.DataFrame({"_dcorr22": d_corr, "_std22": std20}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_dcorr22"] = tmp["_dcorr22"]
    df["_std22"] = tmp["_std22"]

    df["_rk_std22"] = df.groupby("date")["_std22"].transform(cs_rank)
    df["alpha022"] = -1 * df["_dcorr22"] * df["_rk_std22"]
    return _finish(df, idx, "alpha022", append, ["_dcorr22", "_std22", "_rk_std22"])


def add_alpha023(
    df: pd.DataFrame,
    high_col: str = "high",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #023: if mean(high,20) < high: -1*delta(high,2), else: 0

    Required columns: date, code, name, ``high_col``.
    """
    _validate(df, ["date", "code", "name", high_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        m20 = ts_mean(g[high_col], 20)
        d2 = delta(g[high_col], 2)
        return pd.Series(np.where(m20 < g[high_col], -1 * d2, 0.0), index=g.index)

    df["alpha023"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha023", append, [])


def add_alpha024(
    df: pd.DataFrame,
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #024:
        if delta(mean(close,100)/mean(close,100), 100)/delay(close,100) <= 0.05:
            -(close - ts_min(close,100))
        else:
            -1 * delta(close, 3)

    Note: delta(mean(close,100)/mean(close,100), 100) simplifies to delta(1, 100) = 0,
    so the condition is always 0/delay(close,100) <= 0.05, which is True whenever
    delay(close,100) > 0. The practical implementation uses delta(mean(close,100), 100)
    divided by delay(close,100) as the intended formula.

    Required columns: date, code, name, ``close_col``.
    """
    _validate(df, ["date", "code", "name", close_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        m100 = ts_mean(g[close_col], 100)
        ratio = delta(m100, 100) / delay(g[close_col], 100).replace(0, np.nan)
        cond = ratio <= 0.05
        branch1 = -(g[close_col] - ts_min(g[close_col], 100))
        branch2 = -1 * delta(g[close_col], 3)
        return pd.Series(np.where(cond, branch1, branch2), index=g.index)

    df["alpha024"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha024", append, [])


def add_alpha025(
    df: pd.DataFrame,
    returns_col: str = "returns",
    volume_col: str = "volume",
    vwap_col: str = "vwap",
    high_col: str = "high",
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #025: rank(-1 * returns * adv20 * vwap * (high - close))

    Required columns: date, code, name, ``returns_col``, ``volume_col``,
                      ``vwap_col``, ``high_col``, ``close_col``.
    """
    _validate(df, ["date", "code", "name", returns_col, volume_col,
                          vwap_col, high_col, close_col])
    df = _sort(df).copy()
    idx = df.index

    df["_inner25"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: -1 * g[returns_col] * adv(g[volume_col], 20)
               * g[vwap_col] * (g[high_col] - g[close_col]))
        .reset_index(level=0, drop=True)
    )
    df["alpha025"] = df.groupby("date")["_inner25"].transform(cs_rank)
    return _finish(df, idx, "alpha025", append, ["_inner25"])


def add_alpha026(
    df: pd.DataFrame,
    volume_col: str = "volume",
    high_col: str = "high",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #026: -1 * ts_max(correlation(ts_rank(volume,5), ts_rank(high,5), 5), 3)

    Required columns: date, code, name, ``volume_col``, ``high_col``.
    """
    _validate(df, ["date", "code", "name", volume_col, high_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        rv = ts_rank(g[volume_col], 5)
        rh = ts_rank(g[high_col], 5)
        corr = correlation(rv, rh, 5)
        return -1 * ts_max(corr, 3)

    df["alpha026"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: pd.Series(_per_stock(g), index=g.index))
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha026", append, [])


def add_alpha027(
    df: pd.DataFrame,
    volume_col: str = "volume",
    vwap_col: str = "vwap",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #027: if rank(mean(correlation(rank(volume),rank(vwap),6),2)/0.5) > 0.5: -1, else: 1

    Required columns: date, code, name, ``volume_col``, ``vwap_col``.
    """
    _validate(df, ["date", "code", "name", volume_col, vwap_col])
    df = _sort(df).copy()
    idx = df.index

    df["_rv27"] = df.groupby("date")[volume_col].transform(cs_rank)
    df["_rw27"] = df.groupby("date")[vwap_col].transform(cs_rank)

    df["_corr27"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: ts_mean(correlation(g["_rv27"], g["_rw27"], 6), 2) / 0.5)
        .reset_index(level=0, drop=True)
    )
    df["_rk27"] = df.groupby("date")["_corr27"].transform(cs_rank)
    df["alpha027"] = np.where(df["_rk27"] > 0.5, -1.0, 1.0)
    return _finish(df, idx, "alpha027", append, ["_rv27", "_rw27", "_corr27", "_rk27"])


def add_alpha028(
    df: pd.DataFrame,
    volume_col: str = "volume",
    low_col: str = "low",
    high_col: str = "high",
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #028: scale(correlation(adv20, low, 5) + (high+low)/2 - close)

    Required columns: date, code, name, ``volume_col``, ``low_col``, ``high_col``, ``close_col``.
    """
    _validate(df, ["date", "code", "name", volume_col, low_col, high_col, close_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        adv20 = adv(g[volume_col], 20)
        corr = correlation(adv20, g[low_col], 5)
        return corr + (g[high_col] + g[low_col]) / 2 - g[close_col]

    df["_inner28"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: pd.Series(_per_stock(g), index=g.index))
        .reset_index(level=0, drop=True)
    )
    df["alpha028"] = df.groupby("date")["_inner28"].transform(scale_alpha)
    return _finish(df, idx, "alpha028", append, ["_inner28"])


def add_alpha029(
    df: pd.DataFrame,
    close_col: str = "close",
    returns_col: str = "returns",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #029:
        ts_min(ts_product(rank(rank(scale(log(sum(ts_min(rank(rank(-1*rank(delta(close-1,5)))),2),1))))),1),5)
        + ts_rank(delay(-1*returns,6),5)

    Required columns: date, code, name, ``close_col``, ``returns_col``.
    """
    _validate(df, ["date", "code", "name", close_col, returns_col])
    df = _sort(df).copy()
    idx = df.index

    # Compute innermost: -1 * rank(delta(close - 1, 5))
    df["_dc29"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: delta(g[close_col] - 1, 5))
        .reset_index(level=0, drop=True)
    )
    df["_rk1_29"] = -1 * df.groupby("date")["_dc29"].transform(cs_rank)
    df["_rk2_29"] = df.groupby("date")["_rk1_29"].transform(cs_rank)
    df["_rk3_29"] = df.groupby("date")["_rk2_29"].transform(cs_rank)

    # ts_min(rank3, 2) per stock, then sum over window=1 (= identity), log, scale
    df["_tsmin29"] = (
        df.groupby("code")["_rk3_29"]
        .transform(lambda s: ts_min(s, 2))
    )
    # sum(..., 1) is just the value itself; log then scale cross-sectionally
    df["_log29"] = np.log(df["_tsmin29"].clip(lower=1e-10))
    df["_scale29"] = df.groupby("date")["_log29"].transform(scale_alpha)

    # rank(rank(scale(...)))
    df["_rk4_29"] = df.groupby("date")["_scale29"].transform(cs_rank)
    df["_rk5_29"] = df.groupby("date")["_rk4_29"].transform(cs_rank)

    # ts_product(..., 1) = identity; ts_min(..., 5)
    df["_part1_29"] = (
        df.groupby("code")["_rk5_29"]
        .transform(lambda s: ts_min(s, 5))
    )

    # part 2: ts_rank(delay(-1*returns, 6), 5)
    df["_part2_29"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: ts_rank(delay(-1 * g[returns_col], 6), 5))
        .reset_index(level=0, drop=True)
    )

    df["alpha029"] = df["_part1_29"] + df["_part2_29"]
    return _finish(df, idx, "alpha029", append,
                   ["_dc29", "_rk1_29", "_rk2_29", "_rk3_29", "_tsmin29",
                    "_log29", "_scale29", "_rk4_29", "_rk5_29", "_part1_29", "_part2_29"])


def add_alpha030(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #030: (1 - rank(sign(close-delay(close,1)) + sign(delay(close,1)-delay(close,2))
                          + sign(delay(close,2)-delay(close,3))))
                * sum(volume,5) / sum(volume,20)

    Required columns: date, code, name, ``close_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", close_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        c = g[close_col]
        sign_sum = (np.sign(c - delay(c, 1))
                    + np.sign(delay(c, 1) - delay(c, 2))
                    + np.sign(delay(c, 2) - delay(c, 3)))
        sv5 = ts_sum(g[volume_col], 5)
        sv20 = ts_sum(g[volume_col], 20)
        return pd.DataFrame({
            "_ss30": sign_sum,
            "_volr30": sv5 / sv20.replace(0, np.nan),
        }, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_ss30"] = tmp["_ss30"]
    df["_volr30"] = tmp["_volr30"]

    df["_rk30"] = df.groupby("date")["_ss30"].transform(cs_rank)
    df["alpha030"] = (1 - df["_rk30"]) * df["_volr30"]
    return _finish(df, idx, "alpha030", append, ["_ss30", "_volr30", "_rk30"])


def add_alpha031(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    low_col: str = "low",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #031: rank(rank(rank(decay_linear(-1*rank(rank(delta(close,10))),10))))
                + rank(-1*delta(close,3)) + sign(scale(correlation(adv20,low,12)))

    Required columns: date, code, name, ``close_col``, ``volume_col``, ``low_col``.
    """
    _validate(df, ["date", "code", "name", close_col, volume_col, low_col])
    df = _sort(df).copy()
    idx = df.index

    # Part A: inner rank(rank(delta(close,10)))
    df["_dc31"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: delta(g[close_col], 10))
        .reset_index(level=0, drop=True)
    )
    df["_rk1_31"] = df.groupby("date")["_dc31"].transform(cs_rank)
    df["_rk2_31"] = df.groupby("date")["_rk1_31"].transform(cs_rank)

    # decay_linear(-1 * rank2, 10) then triple rank
    df["_dl31"] = (
        df.groupby("code")["_rk2_31"]
        .transform(lambda s: decay_linear(-1 * s, 10))
    )
    df["_rk3_31"] = df.groupby("date")["_dl31"].transform(cs_rank)
    df["_rk4_31"] = df.groupby("date")["_rk3_31"].transform(cs_rank)
    df["_rk5_31"] = df.groupby("date")["_rk4_31"].transform(cs_rank)

    # Part B: rank(-1 * delta(close,3))
    df["_dc3_31"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: -1 * delta(g[close_col], 3))
        .reset_index(level=0, drop=True)
    )
    df["_rk_b31"] = df.groupby("date")["_dc3_31"].transform(cs_rank)

    # Part C: sign(scale(correlation(adv20, low, 12)))
    df["_corr31"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: correlation(adv(g[volume_col], 20), g[low_col], 12))
        .reset_index(level=0, drop=True)
    )
    df["_scale31"] = df.groupby("date")["_corr31"].transform(scale_alpha)

    df["alpha031"] = df["_rk5_31"] + df["_rk_b31"] + np.sign(df["_scale31"])
    return _finish(df, idx, "alpha031", append,
                   ["_dc31", "_rk1_31", "_rk2_31", "_dl31", "_rk3_31", "_rk4_31",
                    "_rk5_31", "_dc3_31", "_rk_b31", "_corr31", "_scale31"])


def add_alpha032(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #032: scale(mean(correlation(vwap, delay(close,1), 253), 1))

    Simplified form: scale of the 253-day rolling correlation between vwap and
    delay(close,1) (the sum/mean over window=1 is the value itself).

    Required columns: date, code, name, ``vwap_col``, ``close_col``.
    """
    _validate(df, ["date", "code", "name", vwap_col, close_col])
    df = _sort(df).copy()
    idx = df.index

    df["_corr32"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: correlation(g[vwap_col], delay(g[close_col], 1), 253))
        .reset_index(level=0, drop=True)
    )
    df["alpha032"] = df.groupby("date")["_corr32"].transform(scale_alpha)
    return _finish(df, idx, "alpha032", append, ["_corr32"])


def add_alpha033(
    df: pd.DataFrame,
    open_col: str = "open",
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #033: rank(-1 * (1 - open/close))

    Required columns: date, code, name, ``open_col``, ``close_col``.
    """
    _validate(df, ["date", "code", "name", open_col, close_col])
    df = _sort(df).copy()
    idx = df.index

    df["_inner33"] = -1 * (1 - df[open_col] / df[close_col].replace(0, np.nan))
    df["alpha033"] = df.groupby("date")["_inner33"].transform(cs_rank)
    return _finish(df, idx, "alpha033", append, ["_inner33"])


def add_alpha034(
    df: pd.DataFrame,
    returns_col: str = "returns",
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #034: rank(rank(stddev(returns,2))/stddev(returns,5)) + rank(-1*delta(close,1))

    Required columns: date, code, name, ``returns_col``, ``close_col``.
    """
    _validate(df, ["date", "code", "name", returns_col, close_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        std2 = ts_stddev(g[returns_col], 2)
        std5 = ts_stddev(g[returns_col], 5)
        ratio = std2 / std5.replace(0, np.nan)
        dc1 = -1 * delta(g[close_col], 1)
        return pd.DataFrame({"_ratio34": ratio, "_dc34": dc1}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_ratio34"] = tmp["_ratio34"]
    df["_dc34"] = tmp["_dc34"]

    df["_rk1_34"] = df.groupby("date")["_ratio34"].transform(cs_rank)
    df["_rk2_34"] = df.groupby("date")["_rk1_34"].transform(cs_rank)
    df["_rk3_34"] = df.groupby("date")["_dc34"].transform(cs_rank)

    df["alpha034"] = df["_rk2_34"] + df["_rk3_34"]
    return _finish(df, idx, "alpha034", append,
                   ["_ratio34", "_dc34", "_rk1_34", "_rk2_34", "_rk3_34"])


def add_alpha035(
    df: pd.DataFrame,
    volume_col: str = "volume",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    returns_col: str = "returns",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #035: ts_rank(volume,32) * (1-ts_rank(close+high-low,16)) * (1-ts_rank(returns,32))

    Required columns: date, code, name, ``volume_col``, ``close_col``,
                      ``high_col``, ``low_col``, ``returns_col``.
    """
    _validate(df, ["date", "code", "name", volume_col, close_col,
                          high_col, low_col, returns_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        tv = ts_rank(g[volume_col], 32)
        tc = ts_rank(g[close_col] + g[high_col] - g[low_col], 16)
        tr = ts_rank(g[returns_col], 32)
        return tv * (1 - tc) * (1 - tr)

    df["alpha035"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: pd.Series(_per_stock(g), index=g.index))
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha035", append, [])


def add_alpha036(
    df: pd.DataFrame,
    close_col: str = "close",
    open_col: str = "open",
    volume_col: str = "volume",
    vwap_col: str = "vwap",
    returns_col: str = "returns",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #036:
        2.21 * rank(correlation(close-open, delay(volume,1), 15))
        + 0.7 * rank(open-close)
        + 0.73 * rank(ts_rank(delay(-returns,6), 5))
        + rank(abs(correlation(vwap, adv20, 6)))
        + 0.6 * rank((ts_mean(close,200) - open) / close)

    Required columns: date, code, name, ``close_col``, ``open_col``,
                      ``volume_col``, ``vwap_col``, ``returns_col``.
    """
    _validate(df, ["date", "code", "name", close_col, open_col,
                          volume_col, vwap_col, returns_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        corr15 = correlation(g[close_col] - g[open_col], delay(g[volume_col], 1), 15)
        tsr_ret = ts_rank(delay(-1 * g[returns_col], 6), 5)
        adv20 = adv(g[volume_col], 20)
        corr_vw = correlation(g[vwap_col], adv20, 6).abs()
        m200 = ts_mean(g[close_col], 200)
        ratio = (m200 - g[open_col]) / g[close_col].replace(0, np.nan)
        return pd.DataFrame({
            "_c36a": corr15,
            "_c36b": g[open_col] - g[close_col],
            "_c36c": tsr_ret,
            "_c36d": corr_vw,
            "_c36e": ratio,
        }, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    for col in ["_c36a", "_c36b", "_c36c", "_c36d", "_c36e"]:
        df[col] = tmp[col]

    df["_rk36a"] = df.groupby("date")["_c36a"].transform(cs_rank)
    df["_rk36b"] = df.groupby("date")["_c36b"].transform(cs_rank)
    df["_rk36c"] = df.groupby("date")["_c36c"].transform(cs_rank)
    df["_rk36d"] = df.groupby("date")["_c36d"].transform(cs_rank)
    df["_rk36e"] = df.groupby("date")["_c36e"].transform(cs_rank)

    df["alpha036"] = (2.21 * df["_rk36a"] + 0.7 * df["_rk36b"]
                      + 0.73 * df["_rk36c"] + df["_rk36d"] + 0.6 * df["_rk36e"])
    return _finish(df, idx, "alpha036", append,
                   ["_c36a", "_c36b", "_c36c", "_c36d", "_c36e",
                    "_rk36a", "_rk36b", "_rk36c", "_rk36d", "_rk36e"])


def add_alpha037(
    df: pd.DataFrame,
    open_col: str = "open",
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #037: rank(correlation(delay(open-close,1), close, 200)) + rank(open-close)

    Required columns: date, code, name, ``open_col``, ``close_col``.
    """
    _validate(df, ["date", "code", "name", open_col, close_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        spread = g[open_col] - g[close_col]
        corr200 = correlation(delay(spread, 1), g[close_col], 200)
        return pd.DataFrame({"_corr37": corr200, "_spread37": spread}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_corr37"] = tmp["_corr37"]
    df["_spread37"] = tmp["_spread37"]

    df["_rk37a"] = df.groupby("date")["_corr37"].transform(cs_rank)
    df["_rk37b"] = df.groupby("date")["_spread37"].transform(cs_rank)
    df["alpha037"] = df["_rk37a"] + df["_rk37b"]
    return _finish(df, idx, "alpha037", append, ["_corr37", "_spread37", "_rk37a", "_rk37b"])


def add_alpha038(
    df: pd.DataFrame,
    close_col: str = "close",
    open_col: str = "open",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #038: -1 * rank(ts_rank(close,10)) * rank(close/open)

    Required columns: date, code, name, ``close_col``, ``open_col``.
    """
    _validate(df, ["date", "code", "name", close_col, open_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        tsr = ts_rank(g[close_col], 10)
        ratio = g[close_col] / g[open_col].replace(0, np.nan)
        return pd.DataFrame({"_tsr38": tsr, "_ratio38": ratio}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_tsr38"] = tmp["_tsr38"]
    df["_ratio38"] = tmp["_ratio38"]

    df["_rk38a"] = df.groupby("date")["_tsr38"].transform(cs_rank)
    df["_rk38b"] = df.groupby("date")["_ratio38"].transform(cs_rank)
    df["alpha038"] = -1 * df["_rk38a"] * df["_rk38b"]
    return _finish(df, idx, "alpha038", append, ["_tsr38", "_ratio38", "_rk38a", "_rk38b"])


def add_alpha039(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    returns_col: str = "returns",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #039: -1 * rank(delta(close,7) * (1 - rank(decay_linear(volume/adv20, 9))))
                * (1 + rank(sum(returns, 250)))

    Required columns: date, code, name, ``close_col``, ``volume_col``, ``returns_col``.
    """
    _validate(df, ["date", "code", "name", close_col, volume_col, returns_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        adv20 = adv(g[volume_col], 20)
        vol_ratio = g[volume_col] / adv20.replace(0, np.nan)
        dl9 = decay_linear(vol_ratio, 9)
        dc7 = delta(g[close_col], 7)
        sr250 = ts_sum(g[returns_col], 250)
        return pd.DataFrame({
            "_dl39": dl9,
            "_dc39": dc7,
            "_sr39": sr250,
        }, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_dl39"] = tmp["_dl39"]
    df["_dc39"] = tmp["_dc39"]
    df["_sr39"] = tmp["_sr39"]

    df["_rk_dl39"] = df.groupby("date")["_dl39"].transform(cs_rank)
    df["_inner39"] = df["_dc39"] * (1 - df["_rk_dl39"])
    df["_rk_in39"] = df.groupby("date")["_inner39"].transform(cs_rank)
    df["_rk_sr39"] = df.groupby("date")["_sr39"].transform(cs_rank)

    df["alpha039"] = -1 * df["_rk_in39"] * (1 + df["_rk_sr39"])
    return _finish(df, idx, "alpha039", append,
                   ["_dl39", "_dc39", "_sr39", "_rk_dl39", "_inner39",
                    "_rk_in39", "_rk_sr39"])


def add_alpha040(
    df: pd.DataFrame,
    high_col: str = "high",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #040: -1 * rank(stddev(high,10)) * correlation(high, volume, 10)

    Required columns: date, code, name, ``high_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", high_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        std10 = ts_stddev(g[high_col], 10)
        corr10 = correlation(g[high_col], g[volume_col], 10)
        return pd.DataFrame({"_std40": std10, "_corr40": corr10}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_std40"] = tmp["_std40"]
    df["_corr40"] = tmp["_corr40"]

    df["_rk40"] = df.groupby("date")["_std40"].transform(cs_rank)
    df["alpha040"] = -1 * df["_rk40"] * df["_corr40"]
    return _finish(df, idx, "alpha040", append, ["_std40", "_corr40", "_rk40"])
