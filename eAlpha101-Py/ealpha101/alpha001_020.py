"""Alpha #001 – #020"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ._base import _validate, _sort, _finish
from .utils import (
    cs_rank, scale_alpha, ts_rank, ts_mean, ts_stddev, ts_argmax, ts_argmin,
    ts_sum, ts_max, ts_min, ts_product, delta, delay, correlation,
    covariance, decay_linear, signedpower, adv,
)


def add_alpha001(
    mkt_data: pd.DataFrame,
    close_col: str = "close",
    returns_col: str = "returns",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #001: -1 * rank(rank(delta(rank(close),1)) != rank(returns)) * rank(returns)

    Simplified canonical form:
        rank(ts_argmax(signedpower(if(returns < 0, stddev(returns,20), close), 2), 5)) - 0.5

    Required columns: date, code, name, ``close_col``, ``returns_col``.
    """
    _validate(mkt_data, ["date", "code", "name", close_col, returns_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    def _per_stock(g):
        cond = g[returns_col] < 0
        base = np.where(cond, ts_stddev(g[returns_col], 20), g[close_col])
        base = pd.Series(base, index=g.index)
        sp = signedpower(base, 2)
        return ts_argmax(sp, 5)

    df["_arg1"] = df.groupby("code", group_keys=False).apply(_per_stock)

    df["alpha001"] = (
        df.groupby("date")["_arg1"]
        .transform(cs_rank)
        - 0.5
    )
    return _finish(df, idx, "alpha001", append, ["_arg1"])


def add_alpha002(
    mkt_data: pd.DataFrame,
    open_col: str = "open",
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #002: -1 * correlation(rank(delta(log(volume),2)), rank((close-open)/open), 6)

    Required columns: date, code, name, ``open_col``, ``close_col``, ``volume_col``.
    """
    _validate(mkt_data, ["date", "code", "name", open_col, close_col, volume_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    def _per_stock(g):
        x = delta(np.log(g[volume_col].clip(lower=1e-10)), 2)
        y = (g[close_col] - g[open_col]) / g[open_col].replace(0, np.nan)
        # cross-sectional ranks need date-level transform; compute raw for now
        return pd.DataFrame({"_x2": x, "_y2": y}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_x2"] = tmp["_x2"]
    df["_y2"] = tmp["_y2"]

    df["_rx2"] = df.groupby("date")["_x2"].transform(cs_rank)
    df["_ry2"] = df.groupby("date")["_y2"].transform(cs_rank)

    df["alpha002"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: -1 * correlation(g["_rx2"], g["_ry2"], 6))
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha002", append, ["_x2", "_y2", "_rx2", "_ry2"])


def add_alpha003(
    mkt_data: pd.DataFrame,
    open_col: str = "open",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #003: -1 * correlation(rank(open), rank(volume), 10)

    Required columns: date, code, name, ``open_col``, ``volume_col``.
    """
    _validate(mkt_data, ["date", "code", "name", open_col, volume_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    df["_ro3"] = df.groupby("date")[open_col].transform(cs_rank)
    df["_rv3"] = df.groupby("date")[volume_col].transform(cs_rank)

    df["alpha003"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: -1 * correlation(g["_ro3"], g["_rv3"], 10))
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha003", append, ["_ro3", "_rv3"])


def add_alpha004(
    mkt_data: pd.DataFrame,
    low_col: str = "low",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #004: -1 * ts_rank(rank(low), 9)

    Required columns: date, code, name, ``low_col``.
    """
    _validate(mkt_data, ["date", "code", "name", low_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    df["_rl4"] = df.groupby("date")[low_col].transform(cs_rank)
    df["alpha004"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: -1 * ts_rank(g["_rl4"], 9))
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha004", append, ["_rl4"])


def add_alpha005(
    mkt_data: pd.DataFrame,
    open_col: str = "open",
    close_col: str = "close",
    vwap_col: str = "vwap",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #005: rank(open - ts_mean(vwap,10)) * -1 * abs(rank(close - vwap))

    Required columns: date, code, name, ``open_col``, ``close_col``, ``vwap_col``.
    """
    _validate(mkt_data, ["date", "code", "name", open_col, close_col, vwap_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    df["_vwap10"] = (
        df.groupby("code")[vwap_col]
        .transform(lambda s: ts_sum(s, 10) / 10)
    )
    df["_in5a"] = df[open_col] - df["_vwap10"]
    df["_in5b"] = df[close_col] - df[vwap_col]

    df["_rk5a"] = df.groupby("date")["_in5a"].transform(cs_rank)
    df["_rk5b"] = df.groupby("date")["_in5b"].transform(cs_rank)

    df["alpha005"] = df["_rk5a"] * -1 * df["_rk5b"].abs()
    return _finish(df, idx, "alpha005", append, ["_vwap10", "_in5a", "_in5b", "_rk5a", "_rk5b"])


def add_alpha006(
    mkt_data: pd.DataFrame,
    open_col: str = "open",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #006: -1 * correlation(open, volume, 10)

    Required columns: date, code, name, ``open_col``, ``volume_col``.
    """
    _validate(mkt_data, ["date", "code", "name", open_col, volume_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    df["alpha006"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: -1 * correlation(g[open_col], g[volume_col], 10))
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha006", append, [])


def add_alpha007(
    mkt_data: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #007:
        if adv20 < volume:
            -1 * ts_rank(abs(delta(close,7)), 60) * sign(delta(close,7))
        else: -1

    Required columns: date, code, name, ``close_col``, ``volume_col``.
    """
    _validate(mkt_data, ["date", "code", "name", close_col, volume_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    def _per_stock(g):
        adv20 = adv(g[volume_col], 20)
        d7 = delta(g[close_col], 7)
        tsr = ts_rank(d7.abs(), 60)
        inner = -1 * tsr * np.sign(d7)
        return inner.where(adv20 < g[volume_col], -1)

    df["alpha007"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha007", append, [])


def add_alpha008(
    mkt_data: pd.DataFrame,
    open_col: str = "open",
    returns_col: str = "returns",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #008: -1 * rank(sum(open,5)*sum(returns,5) - delay(sum(open,5)*sum(returns,5),10))

    Required columns: date, code, name, ``open_col``, ``returns_col``.
    """
    _validate(mkt_data, ["date", "code", "name", open_col, returns_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    def _per_stock(g):
        prod = ts_sum(g[open_col], 5) * ts_sum(g[returns_col], 5)
        return prod - delay(prod, 10)

    df["_diff8"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )
    df["alpha008"] = -1 * df.groupby("date")["_diff8"].transform(cs_rank)
    return _finish(df, idx, "alpha008", append, ["_diff8"])


def add_alpha009(
    mkt_data: pd.DataFrame,
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #009:
        if ts_min(delta(close,1),5) > 0: delta(close,1)
        elif ts_max(delta(close,1),5) < 0: delta(close,1)
        else: -1*delta(close,1)

    Required columns: date, code, name, ``close_col``.
    """
    _validate(mkt_data, ["date", "code", "name", close_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    def _per_stock(g):
        d = delta(g[close_col], 1)
        mn = ts_min(d, 5)
        mx = ts_max(d, 5)
        return np.where(mn > 0, d, np.where(mx < 0, d, -1 * d))

    df["alpha009"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: pd.Series(_per_stock(g), index=g.index))
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha009", append, [])


def add_alpha010(
    mkt_data: pd.DataFrame,
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #010: rank(if ts_min(delta(close,1),4) > 0: delta(close,1)
                     elif ts_max(delta(close,1),4) < 0: delta(close,1)
                     else: -1*delta(close,1))

    Required columns: date, code, name, ``close_col``.
    """
    _validate(mkt_data, ["date", "code", "name", close_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    def _per_stock(g):
        d = delta(g[close_col], 1)
        mn = ts_min(d, 4)
        mx = ts_max(d, 4)
        return np.where(mn > 0, d, np.where(mx < 0, d, -1 * d))

    df["_in10"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: pd.Series(_per_stock(g), index=g.index))
        .reset_index(level=0, drop=True)
    )
    df["alpha010"] = df.groupby("date")["_in10"].transform(cs_rank)
    return _finish(df, idx, "alpha010", append, ["_in10"])


def add_alpha011(
    mkt_data: pd.DataFrame,
    vwap_col: str = "vwap",
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #011: (rank(ts_max(vwap-close,3)) + rank(ts_min(vwap-close,3))) * rank(delta(volume,3))

    Required columns: date, code, name, ``vwap_col``, ``close_col``, ``volume_col``.
    """
    _validate(mkt_data, ["date", "code", "name", vwap_col, close_col, volume_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    def _per_stock(g):
        spread = g[vwap_col] - g[close_col]
        return pd.DataFrame({
            "_tsmax11": ts_max(spread, 3),
            "_tsmin11": ts_min(spread, 3),
            "_dvol11": delta(g[volume_col], 3),
        }, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_tsmax11"] = tmp["_tsmax11"]
    df["_tsmin11"] = tmp["_tsmin11"]
    df["_dvol11"] = tmp["_dvol11"]

    df["_rk_max11"] = df.groupby("date")["_tsmax11"].transform(cs_rank)
    df["_rk_min11"] = df.groupby("date")["_tsmin11"].transform(cs_rank)
    df["_rk_dv11"] = df.groupby("date")["_dvol11"].transform(cs_rank)

    df["alpha011"] = (df["_rk_max11"] + df["_rk_min11"]) * df["_rk_dv11"]
    return _finish(df, idx, "alpha011", append,
                   ["_tsmax11", "_tsmin11", "_dvol11", "_rk_max11", "_rk_min11", "_rk_dv11"])


def add_alpha012(
    mkt_data: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #012: sign(delta(volume,1)) * (-1 * delta(close,1))

    Required columns: date, code, name, ``close_col``, ``volume_col``.
    """
    _validate(mkt_data, ["date", "code", "name", close_col, volume_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    def _per_stock(g):
        return np.sign(delta(g[volume_col], 1)) * (-1 * delta(g[close_col], 1))

    df["alpha012"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: pd.Series(_per_stock(g), index=g.index))
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha012", append, [])


def add_alpha013(
    mkt_data: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #013: -1 * rank(covariance(rank(close), rank(volume), 5))

    Required columns: date, code, name, ``close_col``, ``volume_col``.
    """
    _validate(mkt_data, ["date", "code", "name", close_col, volume_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    df["_rc13"] = df.groupby("date")[close_col].transform(cs_rank)
    df["_rv13"] = df.groupby("date")[volume_col].transform(cs_rank)

    df["_cov13"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: covariance(g["_rc13"], g["_rv13"], 5))
        .reset_index(level=0, drop=True)
    )
    df["alpha013"] = -1 * df.groupby("date")["_cov13"].transform(cs_rank)
    return _finish(df, idx, "alpha013", append, ["_rc13", "_rv13", "_cov13"])


def add_alpha014(
    mkt_data: pd.DataFrame,
    open_col: str = "open",
    returns_col: str = "returns",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #014: (-1 * rank(delta(returns,3))) * correlation(open, volume, 10)

    Required columns: date, code, name, ``open_col``, ``returns_col``, ``volume_col``.
    """
    _validate(mkt_data, ["date", "code", "name", open_col, returns_col, volume_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    def _per_stock(g):
        dr = delta(g[returns_col], 3)
        corr = correlation(g[open_col], g[volume_col], 10)
        return pd.DataFrame({"_dr14": dr, "_corr14": corr}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_dr14"] = tmp["_dr14"]
    df["_corr14"] = tmp["_corr14"]

    df["_rk_dr14"] = df.groupby("date")["_dr14"].transform(cs_rank)
    df["alpha014"] = -1 * df["_rk_dr14"] * df["_corr14"]
    return _finish(df, idx, "alpha014", append, ["_dr14", "_corr14", "_rk_dr14"])


def add_alpha015(
    mkt_data: pd.DataFrame,
    high_col: str = "high",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #015: -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)

    Required columns: date, code, name, ``high_col``, ``volume_col``.
    """
    _validate(mkt_data, ["date", "code", "name", high_col, volume_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    df["_rh15"] = df.groupby("date")[high_col].transform(cs_rank)
    df["_rv15"] = df.groupby("date")[volume_col].transform(cs_rank)

    df["_corr15"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: correlation(g["_rh15"], g["_rv15"], 3))
        .reset_index(level=0, drop=True)
    )
    df["_rk_corr15"] = df.groupby("date")["_corr15"].transform(cs_rank)
    df["alpha015"] = (
        -1 * df.groupby("code")["_rk_corr15"]
        .transform(lambda s: ts_sum(s, 3))
    )
    return _finish(df, idx, "alpha015", append, ["_rh15", "_rv15", "_corr15", "_rk_corr15"])


def add_alpha016(
    mkt_data: pd.DataFrame,
    high_col: str = "high",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #016: -1 * rank(covariance(rank(high), rank(volume), 5))

    Required columns: date, code, name, ``high_col``, ``volume_col``.
    """
    _validate(mkt_data, ["date", "code", "name", high_col, volume_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    df["_rh16"] = df.groupby("date")[high_col].transform(cs_rank)
    df["_rv16"] = df.groupby("date")[volume_col].transform(cs_rank)

    df["_cov16"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: covariance(g["_rh16"], g["_rv16"], 5))
        .reset_index(level=0, drop=True)
    )
    df["alpha016"] = -1 * df.groupby("date")["_cov16"].transform(cs_rank)
    return _finish(df, idx, "alpha016", append, ["_rh16", "_rv16", "_cov16"])


def add_alpha017(
    mkt_data: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #017: (-1 * rank(ts_rank(close,10))) * rank(delta(delta(close,1),1))
                * rank(ts_rank(volume/adv20, 5))

    Required columns: date, code, name, ``close_col``, ``volume_col``.
    """
    _validate(mkt_data, ["date", "code", "name", close_col, volume_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    def _per_stock(g):
        tsr_c = ts_rank(g[close_col], 10)
        dd = delta(delta(g[close_col], 1), 1)
        adv20 = adv(g[volume_col], 20)
        tsr_v = ts_rank(g[volume_col] / adv20.replace(0, np.nan), 5)
        return pd.DataFrame({
            "_tsr_c17": tsr_c,
            "_dd17": dd,
            "_tsr_v17": tsr_v,
        }, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_tsr_c17"] = tmp["_tsr_c17"]
    df["_dd17"] = tmp["_dd17"]
    df["_tsr_v17"] = tmp["_tsr_v17"]

    df["_rk1_17"] = df.groupby("date")["_tsr_c17"].transform(cs_rank)
    df["_rk2_17"] = df.groupby("date")["_dd17"].transform(cs_rank)
    df["_rk3_17"] = df.groupby("date")["_tsr_v17"].transform(cs_rank)

    df["alpha017"] = -1 * df["_rk1_17"] * df["_rk2_17"] * df["_rk3_17"]
    return _finish(df, idx, "alpha017", append,
                   ["_tsr_c17", "_dd17", "_tsr_v17", "_rk1_17", "_rk2_17", "_rk3_17"])


def add_alpha018(
    mkt_data: pd.DataFrame,
    close_col: str = "close",
    open_col: str = "open",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #018: -1 * rank(stddev(abs(close-open),5) + (close-open) + correlation(close,open,10))

    Required columns: date, code, name, ``close_col``, ``open_col``.
    """
    _validate(mkt_data, ["date", "code", "name", close_col, open_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    def _per_stock(g):
        diff = g[close_col] - g[open_col]
        std5 = ts_stddev(diff.abs(), 5)
        corr10 = correlation(g[close_col], g[open_col], 10)
        return std5 + diff + corr10

    df["_inner18"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: pd.Series(_per_stock(g), index=g.index))
        .reset_index(level=0, drop=True)
    )
    df["alpha018"] = -1 * df.groupby("date")["_inner18"].transform(cs_rank)
    return _finish(df, idx, "alpha018", append, ["_inner18"])


def add_alpha019(
    mkt_data: pd.DataFrame,
    close_col: str = "close",
    returns_col: str = "returns",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #019: (-1 * sign(close - delay(close,7) + delta(close,7)))
                * (1 + rank(1 + sum(returns,250)))

    Required columns: date, code, name, ``close_col``, ``returns_col``.
    """
    _validate(mkt_data, ["date", "code", "name", close_col, returns_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    def _per_stock(g):
        d7 = delta(g[close_col], 7)
        sign_part = np.sign(g[close_col] - delay(g[close_col], 7) + d7)
        sum_ret = ts_sum(g[returns_col], 250)
        return pd.DataFrame({
            "_sign19": sign_part,
            "_sumr19": sum_ret,
        }, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_sign19"] = tmp["_sign19"]
    df["_sumr19"] = tmp["_sumr19"]

    df["_rk19"] = df.groupby("date")["_sumr19"].transform(lambda s: cs_rank(1 + s))
    df["alpha019"] = -1 * df["_sign19"] * (1 + df["_rk19"])
    return _finish(df, idx, "alpha019", append, ["_sign19", "_sumr19", "_rk19"])


def add_alpha020(
    mkt_data: pd.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #020: (-1 * rank(open - delay(high,1))) * rank(open - delay(close,1))
                * rank(open - delay(low,1))

    Required columns: date, code, name, ``open_col``, ``high_col``, ``low_col``, ``close_col``.
    """
    _validate(mkt_data, ["date", "code", "name", open_col, high_col, low_col, close_col])
    df = _sort(mkt_data).copy()
    idx = mkt_data.index

    def _per_stock(g):
        return pd.DataFrame({
            "_a20": g[open_col] - delay(g[high_col], 1),
            "_b20": g[open_col] - delay(g[close_col], 1),
            "_c20": g[open_col] - delay(g[low_col], 1),
        }, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_a20"] = tmp["_a20"]
    df["_b20"] = tmp["_b20"]
    df["_c20"] = tmp["_c20"]

    df["_rk_a20"] = df.groupby("date")["_a20"].transform(cs_rank)
    df["_rk_b20"] = df.groupby("date")["_b20"].transform(cs_rank)
    df["_rk_c20"] = df.groupby("date")["_c20"].transform(cs_rank)

    df["alpha020"] = -1 * df["_rk_a20"] * df["_rk_b20"] * df["_rk_c20"]
    return _finish(df, idx, "alpha020", append,
                   ["_a20", "_b20", "_c20", "_rk_a20", "_rk_b20", "_rk_c20"])
