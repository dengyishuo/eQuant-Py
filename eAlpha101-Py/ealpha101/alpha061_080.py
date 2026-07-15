"""Alpha #061 – #080"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ._base import _validate, _sort, _finish
from .utils import (
    cs_rank, scale_alpha, ts_rank, ts_mean, ts_stddev, ts_argmax, ts_argmin,
    ts_sum, ts_max, ts_min, ts_product, delta, delay, correlation,
    covariance, decay_linear, signedpower, adv,
)


def add_alpha061(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #061:
        (rank(vwap - ts_min(vwap,16)) < rank(correlation(vwap, adv180, 17)))
        Returns 1 where True, 0 where False.

    Required columns: date, code, name, ``vwap_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", vwap_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        a180 = adv(g[volume_col], 180)
        vwap_diff = g[vwap_col] - ts_min(g[vwap_col], 16)
        corr17 = correlation(g[vwap_col], a180, 17)
        return pd.DataFrame({"_vd61": vwap_diff, "_cr61": corr17}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_vd61"] = tmp["_vd61"]
    df["_cr61"] = tmp["_cr61"]

    df["_rk_vd61"] = df.groupby("date")["_vd61"].transform(cs_rank)
    df["_rk_cr61"] = df.groupby("date")["_cr61"].transform(cs_rank)

    df["alpha061"] = (df["_rk_vd61"] < df["_rk_cr61"]).astype(int)
    return _finish(df, idx, "alpha061", append, ["_vd61", "_cr61", "_rk_vd61", "_rk_cr61"])


def add_alpha062(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #062:
        (rank(correlation(vwap, sum(adv20,22), 9))
         < rank((rank(open)+rank(open)) < (rank((high+low)/2)+rank(high)))) * -1
        Returns -1 where condition True, 0 otherwise.

    Required columns: date, code, name, ``vwap_col``, ``volume_col``,
    ``open_col``, ``high_col``, ``low_col``.
    """
    _validate(df, ["date", "code", "name", vwap_col, volume_col, open_col, high_col, low_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        a20 = adv(g[volume_col], 20)
        sum_adv20_22 = ts_sum(a20, 22)
        corr9 = correlation(g[vwap_col], sum_adv20_22, 9)
        return corr9

    df["_corr62"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )

    df["_rk_open62"] = df.groupby("date")[open_col].transform(cs_rank)
    df["_rk_hl62"] = df.groupby("date").apply(
        lambda g: cs_rank((g[high_col] + g[low_col]) / 2)
    ).reset_index(level=0, drop=True)
    df["_rk_high62"] = df.groupby("date")[high_col].transform(cs_rank)
    df["_rk_corr62"] = df.groupby("date")["_corr62"].transform(cs_rank)

    lhs = df["_rk_open62"] + df["_rk_open62"]
    rhs = df["_rk_hl62"] + df["_rk_high62"]
    df["_inner62"] = (lhs < rhs).astype(float)
    df["_rk_inner62"] = df.groupby("date")["_inner62"].transform(cs_rank)

    df["alpha062"] = -1 * (df["_rk_corr62"] < df["_rk_inner62"]).astype(float)
    tmp_cols = ["_corr62", "_rk_open62", "_rk_hl62", "_rk_high62",
                "_rk_corr62", "_inner62", "_rk_inner62"]
    return _finish(df, idx, "alpha062", append, tmp_cols)


def add_alpha063(
    df: pd.DataFrame,
    neut_close_col: str = "neut_close",
    vwap_col: str = "vwap",
    open_col: str = "open",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #063:
        ts_max(rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2), 8)), 6)
        - rank(decay_linear(
            correlation((vwap*0.318108 + open*(1-0.318108)), sum(adv180,37), 13), 12))

    Industry neutralization of close must be done externally. Pass the
    pre-neutralized close series via ``neut_close_col``.

    Required columns: date, code, name, ``neut_close_col``, ``vwap_col``,
    ``open_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", neut_close_col, vwap_col, open_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    w = 0.318108

    def _per_stock(g):
        dl8 = decay_linear(delta(g[neut_close_col], 2), 8)
        a180 = adv(g[volume_col], 180)
        wvap = g[vwap_col] * w + g[open_col] * (1 - w)
        corr13 = correlation(wvap, ts_sum(a180, 37), 13)
        dl12 = decay_linear(corr13, 12)
        return pd.DataFrame({"_dl8_63": dl8, "_dl12_63": dl12}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_dl8_63"] = tmp["_dl8_63"]
    df["_dl12_63"] = tmp["_dl12_63"]

    df["_rk_dl8_63"] = df.groupby("date")["_dl8_63"].transform(cs_rank)
    df["_rk_dl12_63"] = df.groupby("date")["_dl12_63"].transform(cs_rank)

    df["_tsmax63"] = (
        df.groupby("code")["_rk_dl8_63"]
        .transform(lambda s: ts_max(s, 6))
    )

    df["alpha063"] = df["_tsmax63"] - df["_rk_dl12_63"]
    tmp_cols = ["_dl8_63", "_dl12_63", "_rk_dl8_63", "_rk_dl12_63", "_tsmax63"]
    return _finish(df, idx, "alpha063", append, tmp_cols)


def add_alpha064(
    df: pd.DataFrame,
    open_col: str = "open",
    low_col: str = "low",
    high_col: str = "high",
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #064:
        (rank(correlation(sum(open*0.178404 + low*(1-0.178404), 12),
                          sum(adv120, 12), 16))
         < rank(delta((high+low)/2*0.178404 + vwap*(1-0.178404), 3))) * -1
        Returns -1 where True, 0 otherwise.

    Required columns: date, code, name, ``open_col``, ``low_col``,
    ``high_col``, ``vwap_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", open_col, low_col, high_col, vwap_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    w = 0.178404

    def _per_stock(g):
        wol = g[open_col] * w + g[low_col] * (1 - w)
        sum_wol12 = ts_sum(wol, 12)
        a120 = adv(g[volume_col], 120)
        sum_a120_12 = ts_sum(a120, 12)
        corr16 = correlation(sum_wol12, sum_a120_12, 16)
        hl_mid = (g[high_col] + g[low_col]) / 2.0
        whlv = hl_mid * w + g[vwap_col] * (1 - w)
        d3 = delta(whlv, 3)
        return pd.DataFrame({"_corr64": corr16, "_delta64": d3}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_corr64"] = tmp["_corr64"]
    df["_delta64"] = tmp["_delta64"]

    df["_rk_corr64"] = df.groupby("date")["_corr64"].transform(cs_rank)
    df["_rk_delta64"] = df.groupby("date")["_delta64"].transform(cs_rank)

    df["alpha064"] = -1 * (df["_rk_corr64"] < df["_rk_delta64"]).astype(float)
    return _finish(df, idx, "alpha064", append,
                   ["_corr64", "_delta64", "_rk_corr64", "_rk_delta64"])


def add_alpha065(
    df: pd.DataFrame,
    open_col: str = "open",
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #065:
        (rank(correlation(open*0.00817205 + vwap*(1-0.00817205),
                          sum(adv60, 8), 6))
         < rank(open - ts_min(open, 13))) * -1
        Returns -1 where True, 0 otherwise.

    Required columns: date, code, name, ``open_col``, ``vwap_col``,
    ``volume_col``.
    """
    _validate(df, ["date", "code", "name", open_col, vwap_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    w = 0.00817205

    def _per_stock(g):
        wov = g[open_col] * w + g[vwap_col] * (1 - w)
        a60 = adv(g[volume_col], 60)
        sum_a60_8 = ts_sum(a60, 8)
        corr6 = correlation(wov, sum_a60_8, 6)
        open_diff = g[open_col] - ts_min(g[open_col], 13)
        return pd.DataFrame({"_corr65": corr6, "_od65": open_diff}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_corr65"] = tmp["_corr65"]
    df["_od65"] = tmp["_od65"]

    df["_rk_corr65"] = df.groupby("date")["_corr65"].transform(cs_rank)
    df["_rk_od65"] = df.groupby("date")["_od65"].transform(cs_rank)

    df["alpha065"] = -1 * (df["_rk_corr65"] < df["_rk_od65"]).astype(float)
    return _finish(df, idx, "alpha065", append,
                   ["_corr65", "_od65", "_rk_corr65", "_rk_od65"])


def add_alpha066(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    close_col: str = "close",
    open_col: str = "open",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #066:
        rank(decay_linear(delta(vwap, 3), 7))
        + ts_rank(decay_linear((close - open) / vwap, 11), 7)

    Note: simplified from original; uses (close-open)/vwap for the second term.

    Required columns: date, code, name, ``vwap_col``, ``close_col``,
    ``open_col``.
    """
    _validate(df, ["date", "code", "name", vwap_col, close_col, open_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        dl7 = decay_linear(delta(g[vwap_col], 3), 7)
        ratio = (g[close_col] - g[open_col]) / g[vwap_col].replace(0, np.nan)
        dl11 = decay_linear(ratio, 11)
        tsr7 = ts_rank(dl11, 7)
        return pd.DataFrame({"_dl7_66": dl7, "_tsr7_66": tsr7}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_dl7_66"] = tmp["_dl7_66"]
    df["_tsr7_66"] = tmp["_tsr7_66"]

    df["_rk_dl7_66"] = df.groupby("date")["_dl7_66"].transform(cs_rank)
    df["alpha066"] = df["_rk_dl7_66"] + df["_tsr7_66"]
    return _finish(df, idx, "alpha066", append, ["_dl7_66", "_tsr7_66", "_rk_dl7_66"])


def add_alpha067(
    df: pd.DataFrame,
    high_col: str = "high",
    neut_vwap_col: str = "neut_vwap",
    neut_adv20_col: str = "neut_adv20",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #067:
        rank(high - ts_min(high, 2))
        * sqrt(IndNeutralize(
            rank(correlation(IndNeutralize(vwap, IndClass.sector),
                             IndNeutralize(adv20, IndClass.subindustry), 6)),
            IndClass.subindustry))

    Industry neutralization must be performed externally. Pass:
    - ``neut_vwap_col``: vwap neutralized by sector
    - ``neut_adv20_col``: adv20 neutralized by subindustry

    The outer IndNeutralize of the rank(correlation(...)) is omitted as it
    requires group-level metadata not available here; users should apply it
    post-hoc if needed.

    Required columns: date, code, name, ``high_col``, ``neut_vwap_col``,
    ``neut_adv20_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", high_col, neut_vwap_col, neut_adv20_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    df["_hd67"] = (
        df.groupby("code")[high_col]
        .transform(lambda s: s - ts_min(s, 2))
    )

    def _corr_per_stock(g):
        return correlation(g[neut_vwap_col], g[neut_adv20_col], 6)

    df["_corr67"] = (
        df.groupby("code", group_keys=False)
        .apply(_corr_per_stock)
        .reset_index(level=0, drop=True)
    )

    df["_rk_hd67"] = df.groupby("date")["_hd67"].transform(cs_rank)
    df["_rk_corr67"] = df.groupby("date")["_corr67"].transform(cs_rank)

    df["alpha067"] = df["_rk_hd67"] * np.sqrt(df["_rk_corr67"].clip(lower=0))
    return _finish(df, idx, "alpha067", append,
                   ["_hd67", "_corr67", "_rk_hd67", "_rk_corr67"])


def add_alpha068(
    df: pd.DataFrame,
    high_col: str = "high",
    close_col: str = "close",
    low_col: str = "low",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #068:
        (ts_rank(correlation(rank(high), rank(adv15), 8), 13)
         < rank(delta(close*0.518371 + low*(1-0.518371), 1))) * -1
        Returns -1 where True, 0 otherwise.

    Required columns: date, code, name, ``high_col``, ``close_col``,
    ``low_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", high_col, close_col, low_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    w = 0.518371

    df["_rk_high68"] = df.groupby("date")[high_col].transform(cs_rank)

    def _per_stock(g):
        a15 = adv(g[volume_col], 15)
        rk_a15 = a15.rank(pct=True)   # within-stock ts-rank approximation; cs-rank done below
        wcl = g[close_col] * w + g[low_col] * (1 - w)
        d1 = delta(wcl, 1)
        return pd.DataFrame({"_a15_68": a15, "_d1_68": d1}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_a15_68"] = tmp["_a15_68"]
    df["_d1_68"] = tmp["_d1_68"]

    df["_rk_a15_68"] = df.groupby("date")["_a15_68"].transform(cs_rank)
    df["_rk_d1_68"] = df.groupby("date")["_d1_68"].transform(cs_rank)

    def _corr_per_stock(g):
        corr8 = correlation(g["_rk_high68"], g["_rk_a15_68"], 8)
        return ts_rank(corr8, 13)

    df["_tsr_corr68"] = (
        df.groupby("code", group_keys=False)
        .apply(_corr_per_stock)
        .reset_index(level=0, drop=True)
    )

    df["alpha068"] = -1 * (df["_tsr_corr68"] < df["_rk_d1_68"]).astype(float)
    tmp_cols = ["_rk_high68", "_a15_68", "_d1_68", "_rk_a15_68", "_rk_d1_68", "_tsr_corr68"]
    return _finish(df, idx, "alpha068", append, tmp_cols)


def add_alpha069(
    df: pd.DataFrame,
    neut_vwap_col: str = "neut_vwap",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #069:
        rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2), 4))^0.65
        - rank(decay_linear(correlation((high+low)/2, adv20, 8), 7))

    Industry neutralization of vwap must be done externally. Pass the
    pre-neutralized series via ``neut_vwap_col``.

    Required columns: date, code, name, ``neut_vwap_col``, ``high_col``,
    ``low_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", neut_vwap_col, high_col, low_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        tsmax_d2 = ts_max(delta(g[neut_vwap_col], 2), 4)
        a20 = adv(g[volume_col], 20)
        hl_mid = (g[high_col] + g[low_col]) / 2.0
        dl7 = decay_linear(correlation(hl_mid, a20, 8), 7)
        return pd.DataFrame({"_tsmax69": tsmax_d2, "_dl7_69": dl7}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_tsmax69"] = tmp["_tsmax69"]
    df["_dl7_69"] = tmp["_dl7_69"]

    df["_rk_tsmax69"] = df.groupby("date")["_tsmax69"].transform(cs_rank)
    df["_rk_dl7_69"] = df.groupby("date")["_dl7_69"].transform(cs_rank)

    df["alpha069"] = signedpower(df["_rk_tsmax69"], 0.65) - df["_rk_dl7_69"]
    return _finish(df, idx, "alpha069", append,
                   ["_tsmax69", "_dl7_69", "_rk_tsmax69", "_rk_dl7_69"])


def add_alpha070(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    neut_close_col: str = "neut_close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #070:
        rank(delta(vwap, 1))^0.1
        * ts_rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 18), 18)

    Industry neutralization of close must be done externally. Pass the
    pre-neutralized series via ``neut_close_col``.

    Required columns: date, code, name, ``vwap_col``, ``neut_close_col``,
    ``volume_col``.
    """
    _validate(df, ["date", "code", "name", vwap_col, neut_close_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    df["_dv70"] = (
        df.groupby("code")[vwap_col]
        .transform(lambda s: delta(s, 1))
    )
    df["_rk_dv70"] = df.groupby("date")["_dv70"].transform(cs_rank)

    def _per_stock(g):
        a50 = adv(g[volume_col], 50)
        corr18 = correlation(g[neut_close_col], a50, 18)
        tsr18 = ts_rank(corr18, 18)
        return tsr18

    df["_tsr70"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )

    df["alpha070"] = signedpower(df["_rk_dv70"], 0.1) * df["_tsr70"]
    return _finish(df, idx, "alpha070", append, ["_dv70", "_rk_dv70", "_tsr70"])


def add_alpha071(
    df: pd.DataFrame,
    close_col: str = "close",
    low_col: str = "low",
    open_col: str = "open",
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #071:
        max(ts_rank(decay_linear(correlation(ts_rank(close,3), ts_rank(adv180,12), 18), 4), 16),
            ts_rank(decay_linear(rank(low+open-vwap*2)^2, 16), 4))

    Required columns: date, code, name, close, low, open, vwap, volume.
    """
    _validate(df, ["date", "code", "name", close_col, low_col, open_col, vwap_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        adv180 = adv(g[volume_col], 180)
        trc = ts_rank(g[close_col], 3)
        tra = ts_rank(adv180, 12)
        part1 = ts_rank(decay_linear(correlation(trc, tra, 18), 4), 16)

        inner2 = g[low_col] + g[open_col] - g[vwap_col] * 2
        r2 = inner2 ** 2
        part2 = ts_rank(decay_linear(r2, 16), 4)
        return pd.Series(np.maximum(part1.values, part2.values), index=g.index)

    df["alpha071"] = (
        df.groupby("code", group_keys=False).apply(_per_stock)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha071", append, [])


def add_alpha072(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #072:
        rank(decay_linear(correlation((high+low)/2, adv40, 8), 10)) /
        rank(decay_linear(correlation(ts_rank(vwap,3), ts_rank(volume,18), 6), 2))

    Required columns: date, code, name, high, low, vwap, volume.
    """
    _validate(df, ["date", "code", "name", high_col, low_col, vwap_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        mid = (g[high_col] + g[low_col]) / 2
        a40 = adv(g[volume_col], 40)
        num = decay_linear(correlation(mid, a40, 8), 10)
        trv = ts_rank(g[vwap_col], 3)
        trvol = ts_rank(g[volume_col], 18)
        den = decay_linear(correlation(trv, trvol, 6), 2)
        return pd.DataFrame({"_n72": num, "_d72": den}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_n72"] = tmp["_n72"]
    df["_d72"] = tmp["_d72"]
    df["_rn72"] = df.groupby("date")["_n72"].transform(cs_rank)
    df["_rd72"] = df.groupby("date")["_d72"].transform(cs_rank)
    df["alpha072"] = df["_rn72"] / df["_rd72"].replace(0, np.nan)
    return _finish(df, idx, "alpha072", append, ["_n72", "_d72", "_rn72", "_rd72"])


def add_alpha073(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    close_col: str = "close",
    open_col: str = "open",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #073:
        max(rank(decay_linear(delta(vwap,4), 2)),
            ts_rank(decay_linear(-1*delta((close*0.147155+open*(1-0.147155)),2)/
                                  ((close*0.147155+open*(1-0.147155)))+vwap, 3)*-1, 16)) * -1

    Required columns: date, code, name, vwap, close, open.
    """
    _validate(df, ["date", "code", "name", vwap_col, close_col, open_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        dv = decay_linear(delta(g[vwap_col], 4), 2)
        price = g[close_col] * 0.147155 + g[open_col] * (1 - 0.147155)
        dp = delta(price, 2) / price.replace(0, np.nan) + g[vwap_col]
        part2 = ts_rank(decay_linear(-1 * dp, 3) * -1, 16)
        return pd.DataFrame({"_dv73": dv, "_p273": part2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_dv73"] = tmp["_dv73"]
    df["_p273"] = tmp["_p273"]
    df["_rdv73"] = df.groupby("date")["_dv73"].transform(cs_rank)
    df["alpha073"] = -1 * np.maximum(df["_rdv73"].values, df["_p273"].values)
    return _finish(df, idx, "alpha073", append, ["_dv73", "_p273", "_rdv73"])


def add_alpha074(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #074:
        (rank(correlation(close, sum(adv30,37), 15)) <
         rank(correlation(rank(high*0.0261661+vwap*(1-0.0261661)), rank(volume), 11))) * -1

    Required columns: date, code, name, close, high, vwap, volume.
    """
    _validate(df, ["date", "code", "name", close_col, high_col, vwap_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        a30 = adv(g[volume_col], 30)
        c1 = correlation(g[close_col], ts_sum(a30, 37), 15)
        price = g[high_col] * 0.0261661 + g[vwap_col] * (1 - 0.0261661)
        return pd.DataFrame({"_c174": c1, "_pr74": price}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_c174"] = tmp["_c174"]
    df["_pr74"] = tmp["_pr74"]
    df["_rc174"] = df.groupby("date")["_c174"].transform(cs_rank)
    df["_rpr74"] = df.groupby("date")["_pr74"].transform(cs_rank)
    df["_rvol74"] = df.groupby("date")[volume_col].transform(cs_rank)

    def _per_stock2(g):
        return correlation(g["_rpr74"], g["_rvol74"], 11)

    df["_c274"] = df.groupby("code", group_keys=False).apply(_per_stock2).reset_index(level=0, drop=True)
    df["_rc274"] = df.groupby("date")["_c274"].transform(cs_rank)
    df["alpha074"] = ((df["_rc174"] < df["_rc274"]).astype(int) * -1)
    return _finish(df, idx, "alpha074", append, ["_c174", "_pr74", "_rc174", "_rpr74", "_rvol74", "_c274", "_rc274"])


def add_alpha075(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    low_col: str = "low",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #075:
        rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(adv50), 12))

    Returns 1 where condition holds, 0 otherwise.
    Required columns: date, code, name, vwap, low, volume.
    """
    _validate(df, ["date", "code", "name", vwap_col, low_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    df["_rlow75"] = df.groupby("date")[low_col].transform(cs_rank)

    def _per_stock(g):
        a50 = adv(g[volume_col], 50)
        c1 = correlation(g[vwap_col], g[volume_col], 4)
        ra50 = cs_rank(a50)
        c2 = correlation(g["_rlow75"], ra50, 12)
        return pd.DataFrame({"_c175": c1, "_c275": c2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_c175"] = tmp["_c175"]
    df["_c275"] = tmp["_c275"]
    df["_rc175"] = df.groupby("date")["_c175"].transform(cs_rank)
    df["_rc275"] = df.groupby("date")["_c275"].transform(cs_rank)
    df["alpha075"] = (df["_rc175"] < df["_rc275"]).astype(int)
    return _finish(df, idx, "alpha075", append, ["_rlow75", "_c175", "_c275", "_rc175", "_rc275"])


def add_alpha076(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    neut_low_col: str = "neut_low",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #076:
        max(rank(decay_linear(delta(vwap,1), 11)),
            ts_rank(decay_linear(ts_rank(correlation(IndNeutralize(low), adv81, 8), 19), 17), 19)) * -1

    IndNeutralize(low) must be pre-computed and passed as ``neut_low_col``.
    Required columns: date, code, name, vwap, volume, ``neut_low_col``.
    """
    _validate(df, ["date", "code", "name", vwap_col, volume_col, neut_low_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        dv = decay_linear(delta(g[vwap_col], 1), 11)
        a81 = adv(g[volume_col], 81)
        c = correlation(g[neut_low_col], a81, 8)
        part2 = ts_rank(decay_linear(ts_rank(c, 19), 17), 19)
        return pd.DataFrame({"_dv76": dv, "_p276": part2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_dv76"] = tmp["_dv76"]
    df["_p276"] = tmp["_p276"]
    df["_rdv76"] = df.groupby("date")["_dv76"].transform(cs_rank)
    df["alpha076"] = -1 * np.maximum(df["_rdv76"].values, df["_p276"].values)
    return _finish(df, idx, "alpha076", append, ["_dv76", "_p276", "_rdv76"])


def add_alpha077(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #077:
        min(rank(decay_linear((high+low)/2+high-(vwap+high), 20)),
            rank(decay_linear(correlation((high+low)/2, adv40, 3), 6)))

    Required columns: date, code, name, high, low, vwap, volume.
    """
    _validate(df, ["date", "code", "name", high_col, low_col, vwap_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        mid = (g[high_col] + g[low_col]) / 2
        p1 = decay_linear(mid + g[high_col] - (g[vwap_col] + g[high_col]), 20)
        a40 = adv(g[volume_col], 40)
        p2 = decay_linear(correlation(mid, a40, 3), 6)
        return pd.DataFrame({"_p177": p1, "_p277": p2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_p177"] = tmp["_p177"]
    df["_p277"] = tmp["_p277"]
    df["_rp177"] = df.groupby("date")["_p177"].transform(cs_rank)
    df["_rp277"] = df.groupby("date")["_p277"].transform(cs_rank)
    df["alpha077"] = np.minimum(df["_rp177"].values, df["_rp277"].values)
    return _finish(df, idx, "alpha077", append, ["_p177", "_p277", "_rp177", "_rp277"])


def add_alpha078(
    df: pd.DataFrame,
    low_col: str = "low",
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #078:
        rank(correlation(sum(low*0.352233+vwap*(1-0.352233),19), sum(adv40,19), 6)) /
        rank(correlation(rank(vwap), rank(volume), 5))

    Required columns: date, code, name, low, vwap, volume.
    """
    _validate(df, ["date", "code", "name", low_col, vwap_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    df["_rvwap78"] = df.groupby("date")[vwap_col].transform(cs_rank)
    df["_rvol78"] = df.groupby("date")[volume_col].transform(cs_rank)

    def _per_stock(g):
        price = g[low_col] * 0.352233 + g[vwap_col] * (1 - 0.352233)
        a40 = adv(g[volume_col], 40)
        c1 = correlation(ts_sum(price, 19), ts_sum(a40, 19), 6)
        c2 = correlation(g["_rvwap78"], g["_rvol78"], 5)
        return pd.DataFrame({"_c178": c1, "_c278": c2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_c178"] = tmp["_c178"]
    df["_c278"] = tmp["_c278"]
    df["_rc178"] = df.groupby("date")["_c178"].transform(cs_rank)
    df["_rc278"] = df.groupby("date")["_c278"].transform(cs_rank)
    df["alpha078"] = df["_rc178"] / df["_rc278"].replace(0, np.nan)
    return _finish(df, idx, "alpha078", append, ["_rvwap78", "_rvol78", "_c178", "_c278", "_rc178", "_rc278"])


def add_alpha079(
    df: pd.DataFrame,
    neut_price_col: str = "neut_price79",
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #079:
        rank(delta(IndNeutralize(close*0.60733+open*(1-0.60733)), 1)) <
        rank(correlation(ts_rank(vwap,3), ts_rank(adv150,9), 14))

    IndNeutralize must be pre-computed and passed as ``neut_price_col``.
    Returns 1 where condition holds, 0 otherwise.
    Required columns: date, code, name, vwap, volume, ``neut_price_col``.
    """
    _validate(df, ["date", "code", "name", vwap_col, volume_col, neut_price_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        d1 = delta(g[neut_price_col], 1)
        a150 = adv(g[volume_col], 150)
        c2 = correlation(ts_rank(g[vwap_col], 3), ts_rank(a150, 9), 14)
        return pd.DataFrame({"_d179": d1, "_c279": c2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_d179"] = tmp["_d179"]
    df["_c279"] = tmp["_c279"]
    df["_rd179"] = df.groupby("date")["_d179"].transform(cs_rank)
    df["_rc279"] = df.groupby("date")["_c279"].transform(cs_rank)
    df["alpha079"] = (df["_rd179"] < df["_rc279"]).astype(int)
    return _finish(df, idx, "alpha079", append, ["_d179", "_c279", "_rd179", "_rc279"])


def add_alpha080(
    df: pd.DataFrame,
    neut_price_col: str = "neut_price80",
    high_col: str = "high",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #080:
        (rank(sign(delta(IndNeutralize(open*0.868128+high*(1-0.868128)),4)))^2) *
        (rank(correlation(high, adv10, 5))^1)

    IndNeutralize must be pre-computed and passed as ``neut_price_col``.
    Required columns: date, code, name, high, volume, ``neut_price_col``.
    """
    _validate(df, ["date", "code", "name", high_col, volume_col, neut_price_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        s = np.sign(delta(g[neut_price_col], 4))
        a10 = adv(g[volume_col], 10)
        c = correlation(g[high_col], a10, 5)
        return pd.DataFrame({"_s80": s, "_c80": c}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock)
    df["_s80"] = tmp["_s80"]
    df["_c80"] = tmp["_c80"]
    df["_rs80"] = df.groupby("date")["_s80"].transform(cs_rank)
    df["_rc80"] = df.groupby("date")["_c80"].transform(cs_rank)
    df["alpha080"] = (df["_rs80"] ** 2) * df["_rc80"]
    return _finish(df, idx, "alpha080", append, ["_s80", "_c80", "_rs80", "_rc80"])
