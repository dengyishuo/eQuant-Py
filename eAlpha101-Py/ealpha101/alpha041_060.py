"""Alpha #041 – #060"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ._base import _validate, _sort, _finish
from .utils import (
    cs_rank, scale_alpha, ts_rank, ts_mean, ts_stddev, ts_argmax, ts_argmin,
    ts_sum, ts_max, ts_min, ts_product, delta, delay, correlation,
    covariance, decay_linear, signedpower, adv,
)


def add_alpha041(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    vwap_col: str = "vwap",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #041: pow(high * low, 0.5) - vwap

    Required columns: date, code, name, ``high_col``, ``low_col``, ``vwap_col``.
    """
    _validate(df, ["date", "code", "name", high_col, low_col, vwap_col])
    df = _sort(df).copy()
    idx = df.index

    df["alpha041"] = (df[high_col] * df[low_col]).pow(0.5) - df[vwap_col]
    return _finish(df, idx, "alpha041", append, [])


def add_alpha042(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #042: rank(vwap - close) / rank(vwap + close)

    Required columns: date, code, name, ``vwap_col``, ``close_col``.
    """
    _validate(df, ["date", "code", "name", vwap_col, close_col])
    df = _sort(df).copy()
    idx = df.index

    df["_in42a"] = df[vwap_col] - df[close_col]
    df["_in42b"] = df[vwap_col] + df[close_col]
    df["_rk42a"] = df.groupby("date")["_in42a"].transform(cs_rank)
    df["_rk42b"] = df.groupby("date")["_in42b"].transform(cs_rank)
    df["alpha042"] = df["_rk42a"] / df["_rk42b"].replace(0, np.nan)
    return _finish(df, idx, "alpha042", append, ["_in42a", "_in42b", "_rk42a", "_rk42b"])


def add_alpha043(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #043: ts_rank(volume / adv20, 20) * ts_rank(-delta(close, 7), 8)

    Required columns: date, code, name, ``close_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", close_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        adv20 = adv(g[volume_col], 20)
        vol_ratio = g[volume_col] / adv20.replace(0, np.nan)
        part1 = ts_rank(vol_ratio, 20)
        part2 = ts_rank(-delta(g[close_col], 7), 8)
        return part1 * part2

    df["alpha043"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha043", append, [])


def add_alpha044(
    df: pd.DataFrame,
    high_col: str = "high",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #044: -1 * correlation(high, rank(volume), 5)

    Required columns: date, code, name, ``high_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", high_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    df["_rv44"] = df.groupby("date")[volume_col].transform(cs_rank)

    df["alpha044"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: -1 * correlation(g[high_col], g["_rv44"], 5))
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha044", append, ["_rv44"])


def add_alpha045(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #045:
        -1 * (rank(mean(delay(close,5),20)) * correlation(close,volume,2)
               * rank(correlation(sum(close,5), sum(close,20), 2)))

    Required columns: date, code, name, ``close_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", close_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        delayed5 = delay(g[close_col], 5)
        mean_delay = delayed5.rolling(20, min_periods=1).mean()
        corr_cv = correlation(g[close_col], g[volume_col], 2)
        sum5 = ts_sum(g[close_col], 5)
        sum20 = ts_sum(g[close_col], 20)
        corr_sums = correlation(sum5, sum20, 2)
        return pd.DataFrame(
            {"_md45": mean_delay, "_cv45": corr_cv, "_cs45": corr_sums},
            index=g.index,
        )

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_md45"] = tmp["_md45"]
    df["_cv45"] = tmp["_cv45"]
    df["_cs45"] = tmp["_cs45"]

    df["_rmd45"] = df.groupby("date")["_md45"].transform(cs_rank)
    df["_rcs45"] = df.groupby("date")["_cs45"].transform(cs_rank)

    df["alpha045"] = -1 * (df["_rmd45"] * df["_cv45"] * df["_rcs45"])
    return _finish(df, idx, "alpha045", append,
                   ["_md45", "_cv45", "_cs45", "_rmd45", "_rcs45"])


def add_alpha046(
    df: pd.DataFrame,
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #046:
        let slope = (delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10
        if slope > 0.25:  -1
        elif slope < 0:    1
        else:             -1 * (close - delay(close,1))

    Required columns: date, code, name, ``close_col``.
    """
    _validate(df, ["date", "code", "name", close_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        c = g[close_col]
        slope = (delay(c, 20) - delay(c, 10)) / 10.0 - (delay(c, 10) - c) / 10.0
        d1 = delta(c, 1)
        result = np.where(slope > 0.25, -1.0,
                 np.where(slope < 0.0, 1.0,
                 -1.0 * d1))
        return pd.Series(result, index=g.index)

    df["alpha046"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha046", append, [])


def add_alpha047(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    volume_col: str = "volume",
    vwap_col: str = "vwap",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #047:
        (rank(1/close) * volume / adv20 * high * rank(high - close))
        / rank(vwap - delay(vwap, 5) + vwap - close)

    Note: simplified reasonable approximation of the original formula.

    Required columns: date, code, name, ``close_col``, ``high_col``,
    ``volume_col``, ``vwap_col``.
    """
    _validate(df, ["date", "code", "name", close_col, high_col, volume_col, vwap_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        a20 = adv(g[volume_col], 20)
        denom_inner = (g[vwap_col] - delay(g[vwap_col], 5) + g[vwap_col] - g[close_col])
        return pd.DataFrame({
            "_inv47": 1.0 / g[close_col].replace(0, np.nan),
            "_vr47": g[volume_col] / a20.replace(0, np.nan),
            "_hc47": g[high_col] - g[close_col],
            "_denom47": denom_inner,
        }, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    for col in ["_inv47", "_vr47", "_hc47", "_denom47"]:
        df[col] = tmp[col]

    df["_rk_inv47"] = df.groupby("date")["_inv47"].transform(cs_rank)
    df["_rk_hc47"] = df.groupby("date")["_hc47"].transform(cs_rank)
    df["_rk_denom47"] = df.groupby("date")["_denom47"].transform(cs_rank)

    numerator = df["_rk_inv47"] * df["_vr47"] * df[high_col] * df["_rk_hc47"]
    df["alpha047"] = numerator / df["_rk_denom47"].replace(0, np.nan)

    tmp_cols = ["_inv47", "_vr47", "_hc47", "_denom47",
                "_rk_inv47", "_rk_hc47", "_rk_denom47"]
    return _finish(df, idx, "alpha047", append, tmp_cols)


def add_alpha048(
    df: pd.DataFrame,
    neut_close_ret_col: str = "neut_close_ret",
    neut_vwap_ret_col: str = "neut_vwap_ret",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #048:
        IndNeutralize(delta(close,1)/delay(close,1), IndClass.subindustry)
        - IndNeutralize(vwap - delay(vwap,1)/delay(vwap,1), IndClass.subindustry)

    Industry neutralization must be performed externally. Pass the
    pre-neutralized return series directly via ``neut_close_ret_col`` and
    ``neut_vwap_ret_col``.

    Required columns: date, code, name, ``neut_close_ret_col``,
    ``neut_vwap_ret_col``.
    """
    _validate(df, ["date", "code", "name", neut_close_ret_col, neut_vwap_ret_col])
    df = _sort(df).copy()
    idx = df.index

    df["alpha048"] = df[neut_close_ret_col] - df[neut_vwap_ret_col]
    return _finish(df, idx, "alpha048", append, [])


def add_alpha049(
    df: pd.DataFrame,
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #049:
        let slope = (delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10
        if slope >= -0.1:  1
        else:             -1 * (close - delay(close,1))

    Required columns: date, code, name, ``close_col``.
    """
    _validate(df, ["date", "code", "name", close_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        c = g[close_col]
        slope = (delay(c, 20) - delay(c, 10)) / 10.0 - (delay(c, 10) - c) / 10.0
        d1 = delta(c, 1)
        result = np.where(slope >= -0.1, 1.0, -1.0 * d1)
        return pd.Series(result, index=g.index)

    df["alpha049"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha049", append, [])


def add_alpha050(
    df: pd.DataFrame,
    volume_col: str = "volume",
    vwap_col: str = "vwap",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #050: -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)

    Required columns: date, code, name, ``volume_col``, ``vwap_col``.
    """
    _validate(df, ["date", "code", "name", volume_col, vwap_col])
    df = _sort(df).copy()
    idx = df.index

    df["_rv50"] = df.groupby("date")[volume_col].transform(cs_rank)
    df["_rvwap50"] = df.groupby("date")[vwap_col].transform(cs_rank)

    def _per_stock(g):
        corr = correlation(g["_rv50"], g["_rvwap50"], 5)
        rk_corr = corr.rank(pct=True)   # cross-sectional rank is date-level;
        # per original, inner rank is date-level but we approximate with ts_rank
        return -1 * ts_max(rk_corr, 5)

    # The inner rank(correlation(...)) is cross-sectional; compute corr per stock first
    def _corr_per_stock(g):
        return correlation(g["_rv50"], g["_rvwap50"], 5)

    df["_corr50"] = (
        df.groupby("code", group_keys=False)
        .apply(_corr_per_stock)
        .reset_index(level=0, drop=True)
    )

    df["_rk_corr50"] = df.groupby("date")["_corr50"].transform(cs_rank)

    df["alpha050"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: -1 * ts_max(g["_rk_corr50"], 5))
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha050", append, ["_rv50", "_rvwap50", "_corr50", "_rk_corr50"])


def add_alpha051(
    df: pd.DataFrame,
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #051:
        let slope = (delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10
        if slope >= -0.05:  1
        else:              -1 * (close - delay(close,1))

    Required columns: date, code, name, ``close_col``.
    """
    _validate(df, ["date", "code", "name", close_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        c = g[close_col]
        slope = (delay(c, 20) - delay(c, 10)) / 10.0 - (delay(c, 10) - c) / 10.0
        d1 = delta(c, 1)
        result = np.where(slope >= -0.05, 1.0, -1.0 * d1)
        return pd.Series(result, index=g.index)

    df["alpha051"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha051", append, [])


def add_alpha052(
    df: pd.DataFrame,
    low_col: str = "low",
    returns_col: str = "returns",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #052:
        ((-1 * ts_min(low,5)) + delay(ts_min(low,5),5))
        * rank((sum(returns,240) - sum(returns,20)) / 220)
        * ts_rank(volume, 5)

    Required columns: date, code, name, ``low_col``, ``returns_col``,
    ``volume_col``.
    """
    _validate(df, ["date", "code", "name", low_col, returns_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        min_low5 = ts_min(g[low_col], 5)
        part1 = -1 * min_low5 + delay(min_low5, 5)
        ret_diff = (ts_sum(g[returns_col], 240) - ts_sum(g[returns_col], 20)) / 220.0
        part3 = ts_rank(g[volume_col], 5)
        return pd.DataFrame(
            {"_p1_52": part1, "_ret52": ret_diff, "_p3_52": part3},
            index=g.index,
        )

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_p1_52"] = tmp["_p1_52"]
    df["_ret52"] = tmp["_ret52"]
    df["_p3_52"] = tmp["_p3_52"]

    df["_rk_ret52"] = df.groupby("date")["_ret52"].transform(cs_rank)
    df["alpha052"] = df["_p1_52"] * df["_rk_ret52"] * df["_p3_52"]
    return _finish(df, idx, "alpha052", append, ["_p1_52", "_ret52", "_p3_52", "_rk_ret52"])


def add_alpha053(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #053: -1 * delta((1 - (high - close) / (close - low)), 9)

    A small epsilon guards against close == low.

    Required columns: date, code, name, ``high_col``, ``low_col``,
    ``close_col``.
    """
    _validate(df, ["date", "code", "name", high_col, low_col, close_col])
    df = _sort(df).copy()
    idx = df.index

    eps = 1e-10

    def _per_stock(g):
        denom = (g[close_col] - g[low_col]).replace(0, eps)
        ratio = (g[high_col] - g[close_col]) / denom
        inner = 1.0 - ratio
        return -1 * delta(inner, 9)

    df["alpha053"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha053", append, [])


def add_alpha054(
    df: pd.DataFrame,
    low_col: str = "low",
    close_col: str = "close",
    open_col: str = "open",
    high_col: str = "high",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #054: -1 * (low - close) * open^5 / ((close - high) * close^5)

    Zero denominators are replaced with NaN.

    Required columns: date, code, name, ``low_col``, ``close_col``,
    ``open_col``, ``high_col``.
    """
    _validate(df, ["date", "code", "name", low_col, close_col, open_col, high_col])
    df = _sort(df).copy()
    idx = df.index

    numerator = (df[low_col] - df[close_col]) * (df[open_col] ** 5)
    denom = (df[close_col] - df[high_col]) * (df[close_col] ** 5)
    denom = denom.replace(0, np.nan)
    df["alpha054"] = -1 * numerator / denom
    return _finish(df, idx, "alpha054", append, [])


def add_alpha055(
    df: pd.DataFrame,
    close_col: str = "close",
    low_col: str = "low",
    high_col: str = "high",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #055:
        -1 * correlation(
            rank((close - ts_min(low,12)) / (ts_max(high,12) - ts_min(low,12))),
            rank(volume), 6)

    Required columns: date, code, name, ``close_col``, ``low_col``,
    ``high_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", close_col, low_col, high_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        mn12 = ts_min(g[low_col], 12)
        mx12 = ts_max(g[high_col], 12)
        rng = (mx12 - mn12).replace(0, np.nan)
        ratio = (g[close_col] - mn12) / rng
        return ratio

    df["_ratio55"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )

    df["_rk_ratio55"] = df.groupby("date")["_ratio55"].transform(cs_rank)
    df["_rk_vol55"] = df.groupby("date")[volume_col].transform(cs_rank)

    df["alpha055"] = (
        df.groupby("code", group_keys=False)
        .apply(lambda g: -1 * correlation(g["_rk_ratio55"], g["_rk_vol55"], 6))
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha055", append, ["_ratio55", "_rk_ratio55", "_rk_vol55"])


def add_alpha056(
    df: pd.DataFrame,
    returns_col: str = "returns",
    cap_col: str = "cap",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #056: -1 * (rank(sum(returns,10)) < rank(returns * cap))

    Returns 1 when the condition is True (rank lt), else 0, then negated (-1/0).

    Required columns: date, code, name, ``returns_col``, ``cap_col``.
    """
    _validate(df, ["date", "code", "name", returns_col, cap_col])
    df = _sort(df).copy()
    idx = df.index

    df["_sum_ret56"] = (
        df.groupby("code")[returns_col]
        .transform(lambda s: ts_sum(s, 10))
    )
    df["_ret_cap56"] = df[returns_col] * df[cap_col]

    df["_rk_sum56"] = df.groupby("date")["_sum_ret56"].transform(cs_rank)
    df["_rk_rc56"] = df.groupby("date")["_ret_cap56"].transform(cs_rank)

    df["alpha056"] = -1 * (df["_rk_sum56"] < df["_rk_rc56"]).astype(float)
    return _finish(df, idx, "alpha056", append,
                   ["_sum_ret56", "_ret_cap56", "_rk_sum56", "_rk_rc56"])


def add_alpha057(
    df: pd.DataFrame,
    close_col: str = "close",
    vwap_col: str = "vwap",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #057: (close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)

    Required columns: date, code, name, ``close_col``, ``vwap_col``.
    """
    _validate(df, ["date", "code", "name", close_col, vwap_col])
    df = _sort(df).copy()
    idx = df.index

    df["_argmax57"] = (
        df.groupby("code")[close_col]
        .transform(lambda s: ts_argmax(s, 30))
    )
    df["_rk_am57"] = df.groupby("date")["_argmax57"].transform(cs_rank)
    df["_dl57"] = (
        df.groupby("code")["_rk_am57"]
        .transform(lambda s: decay_linear(s, 2))
    )
    df["alpha057"] = (df[close_col] - df[vwap_col]) / df["_dl57"].replace(0, np.nan)
    return _finish(df, idx, "alpha057", append, ["_argmax57", "_rk_am57", "_dl57"])


def add_alpha058(
    df: pd.DataFrame,
    neut_vwap_col: str = "neut_vwap",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #058:
        -1 * ts_rank(decay_linear(correlation(IndNeutralize(vwap), volume, 4), 8), 6)

    Industry neutralization of vwap must be done externally. Pass the
    pre-neutralized series via ``neut_vwap_col``.

    Required columns: date, code, name, ``neut_vwap_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", neut_vwap_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        corr4 = correlation(g[neut_vwap_col], g[volume_col], 4)
        dl8 = decay_linear(corr4, 8)
        return -1 * ts_rank(dl8, 6)

    df["alpha058"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha058", append, [])


def add_alpha059(
    df: pd.DataFrame,
    neut_vwap2_col: str = "neut_vwap2",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #059:
        -1 * ts_rank(decay_linear(
            correlation(IndNeutralize((vwap*0.728317 + vwap*(1-0.728317)),
                                     IndClass.industry), volume, 4), 16), 8)

    Since the weighted vwap reduces to vwap itself, industry neutralization
    must be performed externally. Pass the pre-neutralized series via
    ``neut_vwap2_col``.

    Required columns: date, code, name, ``neut_vwap2_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", neut_vwap2_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        corr4 = correlation(g[neut_vwap2_col], g[volume_col], 4)
        dl16 = decay_linear(corr4, 16)
        return -1 * ts_rank(dl16, 8)

    df["alpha059"] = (
        df.groupby("code", group_keys=False)
        .apply(_per_stock)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha059", append, [])


def add_alpha060(
    df: pd.DataFrame,
    close_col: str = "close",
    low_col: str = "low",
    high_col: str = "high",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #060:
        0 - (1 * (2*scale(rank(((close-low)-(high-close))/(high-low)*volume))
                 - scale(rank(ts_argmax(close, 10)))))

    Required columns: date, code, name, ``close_col``, ``low_col``,
    ``high_col``, ``volume_col``.
    """
    _validate(df, ["date", "code", "name", close_col, low_col, high_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    hl = (df[high_col] - df[low_col]).replace(0, np.nan)
    df["_in60"] = ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col])) / hl * df[volume_col]

    df["_argmax60"] = (
        df.groupby("code")[close_col]
        .transform(lambda s: ts_argmax(s, 10))
    )

    df["_rk_in60"] = df.groupby("date")["_in60"].transform(cs_rank)
    df["_rk_am60"] = df.groupby("date")["_argmax60"].transform(cs_rank)

    df["_sc_in60"] = df.groupby("date")["_rk_in60"].transform(scale_alpha)
    df["_sc_am60"] = df.groupby("date")["_rk_am60"].transform(scale_alpha)

    df["alpha060"] = 0 - (1 * (2 * df["_sc_in60"] - df["_sc_am60"]))
    tmp_cols = ["_in60", "_argmax60", "_rk_in60", "_rk_am60", "_sc_in60", "_sc_am60"]
    return _finish(df, idx, "alpha060", append, tmp_cols)
