"""Alpha #081 – #101"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ._base import _validate, _sort, _finish
from .utils import (
    cs_rank, scale_alpha, ts_rank, ts_mean, ts_stddev, ts_argmax, ts_argmin,
    ts_sum, ts_max, ts_min, ts_product, delta, delay, correlation,
    covariance, decay_linear, signedpower, adv,
)


def add_alpha081(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #081:
        rank(log(product(rank(rank(correlation(vwap, sum(adv10,49), 8))^4), 14.8717)) -
             rank(correlation(rank(vwap), rank(volume), 5)))

    Required columns: date, code, name, vwap, volume.
    """
    _validate(df, ["date", "code", "name", vwap_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    df["_rvwap81"] = df.groupby("date")[vwap_col].transform(cs_rank)
    df["_rvol81"] = df.groupby("date")[volume_col].transform(cs_rank)

    def _per_stock(g):
        a10 = adv(g[volume_col], 10)
        c1 = correlation(g[vwap_col], ts_sum(a10, 49), 8)
        c2 = correlation(g["_rvwap81"], g["_rvol81"], 5)
        return pd.DataFrame({"_c181": c1, "_c281": c2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_c181"] = tmp["_c181"]
    df["_c281"] = tmp["_c281"]
    df["_rc181a"] = df.groupby("date")["_c181"].transform(cs_rank)
    df["_rc181b"] = df.groupby("date")["_rc181a"].transform(cs_rank)

    def _per_stock2(g):
        prod_val = ts_product(g["_rc181b"] ** 4, 15)
        log_prod = np.log(prod_val.clip(lower=1e-10))
        return log_prod - g["_c281"]

    df["_inner81"] = df.groupby("code", group_keys=False).apply(_per_stock2, include_groups=False).reset_index(level=0, drop=True)
    df["alpha081"] = df.groupby("date")["_inner81"].transform(cs_rank)
    return _finish(df, idx, "alpha081", append, ["_rvwap81", "_rvol81", "_c181", "_c281", "_rc181a", "_rc181b", "_inner81"])


def add_alpha082(
    df: pd.DataFrame,
    open_col: str = "open",
    neut_vol_col: str = "neut_volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #082:
        min(rank(decay_linear(delta(open,1),14)),
            ts_rank(decay_linear(correlation(IndNeutralize(volume), open*0.634196+open*(1-0.634196), 17), 6), 6)) * -1

    IndNeutralize(volume) must be pre-computed and passed as ``neut_vol_col``.
    Required columns: date, code, name, open, ``neut_vol_col``.
    """
    _validate(df, ["date", "code", "name", open_col, neut_vol_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        p1 = decay_linear(delta(g[open_col], 1), 14)
        # open * 0.634196 + open * (1-0.634196) = open
        c = correlation(g[neut_vol_col], g[open_col], 17)
        p2 = ts_rank(decay_linear(c, 6), 6)
        return pd.DataFrame({"_p182": p1, "_p282": p2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_p182"] = tmp["_p182"]
    df["_p282"] = tmp["_p282"]
    df["_rp182"] = df.groupby("date")["_p182"].transform(cs_rank)
    df["alpha082"] = -1 * np.minimum(df["_rp182"].values, df["_p282"].values)
    return _finish(df, idx, "alpha082", append, ["_p182", "_p282", "_rp182"])


def add_alpha083(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    vwap_col: str = "vwap",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #083:
        (rank(delay((high-low)/(sum(close,5)/5), 2)) * rank(rank(volume))) /
        (((high-low)/(sum(close,5)/5)) / (vwap-close))

    Required columns: date, code, name, high, low, close, volume, vwap.
    """
    _validate(df, ["date", "code", "name", high_col, low_col, close_col, volume_col, vwap_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        hl_ratio = (g[high_col] - g[low_col]) / (ts_sum(g[close_col], 5) / 5).replace(0, np.nan)
        num_ts = delay(hl_ratio, 2)
        denom_ts = hl_ratio / (g[vwap_col] - g[close_col]).replace(0, np.nan)
        return pd.DataFrame({"_n83": num_ts, "_d83": denom_ts}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_n83"] = tmp["_n83"]
    df["_d83"] = tmp["_d83"]
    df["_rn83"] = df.groupby("date")["_n83"].transform(cs_rank)
    df["_rv83"] = df.groupby("date")[volume_col].transform(cs_rank)
    df["_rrv83"] = df.groupby("date")["_rv83"].transform(cs_rank)
    df["alpha083"] = (df["_rn83"] * df["_rrv83"]) / df["_d83"].replace(0, np.nan)
    return _finish(df, idx, "alpha083", append, ["_n83", "_d83", "_rn83", "_rv83", "_rrv83"])


def add_alpha084(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    close_col: str = "close",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #084: signedpower(ts_rank(vwap - ts_max(vwap,15), 20), delta(close, 4))

    Required columns: date, code, name, vwap, close.
    """
    _validate(df, ["date", "code", "name", vwap_col, close_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        base = ts_rank(g[vwap_col] - ts_max(g[vwap_col], 15), 20)
        exp_ = delta(g[close_col], 4)
        return signedpower(base, exp_)

    df["alpha084"] = (
        df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha084", append, [])


def add_alpha085(
    df: pd.DataFrame,
    high_col: str = "high",
    close_col: str = "close",
    low_col: str = "low",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #085:
        rank(correlation(high*0.876703+close*(1-0.876703), adv30, 9)) ^
        rank(correlation(ts_rank((high+low)/2, 3), ts_rank(volume, 10), 7))

    Required columns: date, code, name, high, close, low, volume.
    """
    _validate(df, ["date", "code", "name", high_col, close_col, low_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        price = g[high_col] * 0.876703 + g[close_col] * (1 - 0.876703)
        a30 = adv(g[volume_col], 30)
        c1 = correlation(price, a30, 9)
        mid = (g[high_col] + g[low_col]) / 2
        c2 = correlation(ts_rank(mid, 3), ts_rank(g[volume_col], 10), 7)
        return pd.DataFrame({"_c185": c1, "_c285": c2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_c185"] = tmp["_c185"]
    df["_c285"] = tmp["_c285"]
    df["_rc185"] = df.groupby("date")["_c185"].transform(cs_rank)
    df["_rc285"] = df.groupby("date")["_c285"].transform(cs_rank)
    df["alpha085"] = df["_rc185"] ** df["_rc285"]
    return _finish(df, idx, "alpha085", append, ["_c185", "_c285", "_rc185", "_rc285"])


def add_alpha086(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #086:
        (ts_rank(correlation(close, sum(adv20,14), 6), 20) <
         rank(5 * rank(rank(close-ts_min(close,14)) / rank(ts_max(close,14)-ts_min(close,14))))) * -1

    Required columns: date, code, name, close, volume.
    """
    _validate(df, ["date", "code", "name", close_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        a20 = adv(g[volume_col], 20)
        trc = ts_rank(correlation(g[close_col], ts_sum(a20, 14), 6), 20)
        mn = ts_min(g[close_col], 14)
        mx = ts_max(g[close_col], 14)
        ratio = (g[close_col] - mn) / (mx - mn).replace(0, np.nan)
        return pd.DataFrame({"_trc86": trc, "_ratio86": ratio}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_trc86"] = tmp["_trc86"]
    df["_ratio86"] = tmp["_ratio86"]
    df["_r186"] = df.groupby("date")["_ratio86"].transform(cs_rank)
    df["_r286"] = df.groupby("date")["_r186"].transform(cs_rank)
    df["_rhs86"] = df.groupby("date")["_r286"].transform(lambda s: cs_rank(5 * s))
    df["alpha086"] = ((df["_trc86"] < df["_rhs86"]).astype(int) * -1)
    return _finish(df, idx, "alpha086", append, ["_trc86", "_ratio86", "_r186", "_r286", "_rhs86"])


def add_alpha087(
    df: pd.DataFrame,
    close_col: str = "close",
    vwap_col: str = "vwap",
    low_col: str = "low",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #087:
        max(rank(decay_linear(delta(close*0.369701+vwap*(1-0.369701), 1), 11)),
            ts_rank(decay_linear(ts_rank(correlation(ts_rank(low,7), ts_rank(adv10,11), 6), 4), 14), 8)) * -1

    Required columns: date, code, name, close, vwap, low, volume.
    """
    _validate(df, ["date", "code", "name", close_col, vwap_col, low_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        price = g[close_col] * 0.369701 + g[vwap_col] * (1 - 0.369701)
        p1 = decay_linear(delta(price, 1), 11)
        a10 = adv(g[volume_col], 10)
        c = correlation(ts_rank(g[low_col], 7), ts_rank(a10, 11), 6)
        p2 = ts_rank(decay_linear(ts_rank(c, 4), 14), 8)
        return pd.DataFrame({"_p187": p1, "_p287": p2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_p187"] = tmp["_p187"]
    df["_p287"] = tmp["_p287"]
    df["_rp187"] = df.groupby("date")["_p187"].transform(cs_rank)
    df["alpha087"] = -1 * np.maximum(df["_rp187"].values, df["_p287"].values)
    return _finish(df, idx, "alpha087", append, ["_p187", "_p287", "_rp187"])


def add_alpha088(
    df: pd.DataFrame,
    open_col: str = "open",
    low_col: str = "low",
    high_col: str = "high",
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #088:
        min(rank(decay_linear((rank(open)+rank(low))-(rank(high)+rank(close)), 8)),
            ts_rank(decay_linear(correlation(ts_rank(close,8), ts_rank(adv60,20), 8), 6), 2))

    Required columns: date, code, name, open, low, high, close, volume.
    """
    _validate(df, ["date", "code", "name", open_col, low_col, high_col, close_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    df["_ro88"] = df.groupby("date")[open_col].transform(cs_rank)
    df["_rl88"] = df.groupby("date")[low_col].transform(cs_rank)
    df["_rh88"] = df.groupby("date")[high_col].transform(cs_rank)
    df["_rc88"] = df.groupby("date")[close_col].transform(cs_rank)
    df["_inner88"] = (df["_ro88"] + df["_rl88"]) - (df["_rh88"] + df["_rc88"])

    def _per_stock(g):
        p1 = decay_linear(g["_inner88"], 8)
        a60 = adv(g[volume_col], 60)
        c = correlation(ts_rank(g[close_col], 8), ts_rank(a60, 20), 8)
        p2 = ts_rank(decay_linear(c, 6), 2)
        return pd.DataFrame({"_p188": p1, "_p288": p2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_p188"] = tmp["_p188"]
    df["_p288"] = tmp["_p288"]
    df["_rp188"] = df.groupby("date")["_p188"].transform(cs_rank)
    df["alpha088"] = np.minimum(df["_rp188"].values, df["_p288"].values)
    return _finish(df, idx, "alpha088", append, ["_ro88", "_rl88", "_rh88", "_rc88", "_inner88", "_p188", "_p288", "_rp188"])


def add_alpha089(
    df: pd.DataFrame,
    low_col: str = "low",
    volume_col: str = "volume",
    neut_vwap_col: str = "neut_vwap",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #089:
        ts_rank(decay_linear(correlation(low, adv10, 6), 2), 6) -
        ts_rank(decay_linear(delta(IndNeutralize(vwap), 3), 13), 10)

    IndNeutralize(vwap) must be pre-computed and passed as ``neut_vwap_col``.
    Required columns: date, code, name, low, volume, ``neut_vwap_col``.
    """
    _validate(df, ["date", "code", "name", low_col, volume_col, neut_vwap_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        a10 = adv(g[volume_col], 10)
        p1 = ts_rank(decay_linear(correlation(g[low_col], a10, 6), 2), 6)
        p2 = ts_rank(decay_linear(delta(g[neut_vwap_col], 3), 13), 10)
        return p1 - p2

    df["alpha089"] = (
        df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
        .reset_index(level=0, drop=True)
    )
    return _finish(df, idx, "alpha089", append, [])


def add_alpha090(
    df: pd.DataFrame,
    close_col: str = "close",
    low_col: str = "low",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #090:
        rank(close - ts_max(close,4)) ^ rank(correlation(adv5, low, 5)) ^ -1

    Required columns: date, code, name, close, low, volume.
    """
    _validate(df, ["date", "code", "name", close_col, low_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        a5 = adv(g[volume_col], 5)
        c = correlation(a5, g[low_col], 5)
        diff = g[close_col] - ts_max(g[close_col], 4)
        return pd.DataFrame({"_diff90": diff, "_c90": c}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_diff90"] = tmp["_diff90"]
    df["_c90"] = tmp["_c90"]
    df["_rd90"] = df.groupby("date")["_diff90"].transform(cs_rank)
    df["_rc90"] = df.groupby("date")["_c90"].transform(cs_rank)
    df["alpha090"] = df["_rd90"] ** (df["_rc90"] ** -1)
    return _finish(df, idx, "alpha090", append, ["_diff90", "_c90", "_rd90", "_rc90"])


def add_alpha091(
    df: pd.DataFrame,
    neut_close_col: str = "neut_close",
    volume_col: str = "volume",
    vwap_col: str = "vwap",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #091:
        ts_rank(decay_linear(decay_linear(correlation(IndNeutralize(close), volume, 9), 6), 9), 13) -
        rank(decay_linear(correlation(vwap, adv30, 4), 5))

    IndNeutralize(close) must be pre-computed and passed as ``neut_close_col``.
    Required columns: date, code, name, volume, vwap, ``neut_close_col``.
    """
    _validate(df, ["date", "code", "name", volume_col, vwap_col, neut_close_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        c1 = correlation(g[neut_close_col], g[volume_col], 9)
        p1 = ts_rank(decay_linear(decay_linear(c1, 6), 9), 13)
        a30 = adv(g[volume_col], 30)
        p2 = decay_linear(correlation(g[vwap_col], a30, 4), 5)
        return pd.DataFrame({"_p191": p1, "_p291": p2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_p191"] = tmp["_p191"]
    df["_p291"] = tmp["_p291"]
    df["_rp291"] = df.groupby("date")["_p291"].transform(cs_rank)
    df["alpha091"] = df["_p191"] - df["_rp291"]
    return _finish(df, idx, "alpha091", append, ["_p191", "_p291", "_rp291"])


def add_alpha092(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    open_col: str = "open",
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #092:
        min(ts_rank(decay_linear(((high+low)/2+close < low+open)*-1, 14), 18),
            ts_rank(decay_linear(rank(correlation(rank(low), rank(adv30), 3) +
                                     rank(close-open) + rank(vwap-close)), 7), 6))

    Required columns: date, code, name, high, low, close, open, vwap, volume.
    """
    _validate(df, ["date", "code", "name", high_col, low_col, close_col, open_col, vwap_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    df["_rl92"] = df.groupby("date")[low_col].transform(cs_rank)
    df["_rco92"] = df.groupby("date").apply(
        lambda g: cs_rank(g[close_col] - g[open_col])
    ).reset_index(level=0, drop=True)
    df["_rvc92"] = df.groupby("date").apply(
        lambda g: cs_rank(g[vwap_col] - g[close_col])
    ).reset_index(level=0, drop=True)

    def _per_stock(g):
        cond = ((g[high_col] + g[low_col]) / 2 + g[close_col] < g[low_col] + g[open_col]).astype(float)
        p1 = ts_rank(decay_linear(-1 * cond, 14), 18)
        a30 = adv(g[volume_col], 30)
        ra30 = cs_rank(a30)
        c = correlation(g["_rl92"], ra30, 3)
        inner = c + g["_rco92"] + g["_rvc92"]
        p2 = ts_rank(decay_linear(inner, 7), 6)
        return pd.DataFrame({"_p192": p1, "_p292": p2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_p192"] = tmp["_p192"]
    df["_p292"] = tmp["_p292"]
    df["alpha092"] = np.minimum(df["_p192"].values, df["_p292"].values)
    return _finish(df, idx, "alpha092", append, ["_rl92", "_rco92", "_rvc92", "_p192", "_p292"])


def add_alpha093(
    df: pd.DataFrame,
    neut_vwap_col: str = "neut_vwap",
    volume_col: str = "volume",
    close_col: str = "close",
    vwap_col: str = "vwap",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #093:
        ts_rank(decay_linear(correlation(IndNeutralize(vwap), adv81, 17), 19), 7) /
        rank(decay_linear(delta(close*0.524434+vwap*(1-0.524434), 3), 6))

    IndNeutralize(vwap) must be pre-computed and passed as ``neut_vwap_col``.
    Required columns: date, code, name, volume, close, vwap, ``neut_vwap_col``.
    """
    _validate(df, ["date", "code", "name", volume_col, close_col, vwap_col, neut_vwap_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        a81 = adv(g[volume_col], 81)
        num = ts_rank(decay_linear(correlation(g[neut_vwap_col], a81, 17), 19), 7)
        price = g[close_col] * 0.524434 + g[vwap_col] * (1 - 0.524434)
        den = decay_linear(delta(price, 3), 6)
        return pd.DataFrame({"_n93": num, "_d93": den}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_n93"] = tmp["_n93"]
    df["_d93"] = tmp["_d93"]
    df["_rd93"] = df.groupby("date")["_d93"].transform(cs_rank)
    df["alpha093"] = df["_n93"] / df["_rd93"].replace(0, np.nan)
    return _finish(df, idx, "alpha093", append, ["_n93", "_d93", "_rd93"])


def add_alpha094(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #094:
        rank(vwap - ts_min(vwap,11)) ^
        ts_rank(correlation(ts_rank(vwap,19), ts_rank(adv60,4), 18), 2)

    Required columns: date, code, name, vwap, volume.
    """
    _validate(df, ["date", "code", "name", vwap_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        a60 = adv(g[volume_col], 60)
        exp_ = ts_rank(correlation(ts_rank(g[vwap_col], 19), ts_rank(a60, 4), 18), 2)
        diff = g[vwap_col] - ts_min(g[vwap_col], 11)
        return pd.DataFrame({"_diff94": diff, "_exp94": exp_}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_diff94"] = tmp["_diff94"]
    df["_exp94"] = tmp["_exp94"]
    df["_rd94"] = df.groupby("date")["_diff94"].transform(cs_rank)
    df["alpha094"] = df["_rd94"] ** df["_exp94"]
    return _finish(df, idx, "alpha094", append, ["_diff94", "_exp94", "_rd94"])


def add_alpha095(
    df: pd.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #095:
        rank(open - ts_min(open,12)) <
        ts_rank(rank(correlation(sum((high+low)/2,19), sum(adv40,19), 12)), 11)

    Returns 1 where condition holds, 0 otherwise.
    Required columns: date, code, name, open, high, low, volume.
    """
    _validate(df, ["date", "code", "name", open_col, high_col, low_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        diff = g[open_col] - ts_min(g[open_col], 12)
        mid = (g[high_col] + g[low_col]) / 2
        a40 = adv(g[volume_col], 40)
        c = correlation(ts_sum(mid, 19), ts_sum(a40, 19), 12)
        return pd.DataFrame({"_diff95": diff, "_c95": c}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_diff95"] = tmp["_diff95"]
    df["_c95"] = tmp["_c95"]
    df["_rd95"] = df.groupby("date")["_diff95"].transform(cs_rank)
    df["_rc95"] = df.groupby("date")["_c95"].transform(cs_rank)

    def _per_stock2(g):
        return ts_rank(g["_rc95"], 11)

    df["_tsr95"] = df.groupby("code", group_keys=False).apply(_per_stock2, include_groups=False).reset_index(level=0, drop=True)
    df["alpha095"] = (df["_rd95"] < df["_tsr95"]).astype(int)
    return _finish(df, idx, "alpha095", append, ["_diff95", "_c95", "_rd95", "_rc95", "_tsr95"])


def add_alpha096(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    close_col: str = "close",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #096:
        max(ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 3), 4), 8),
            ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close,7), ts_rank(adv60,4), 4), 12), 14), 13)) * -1

    Required columns: date, code, name, vwap, close, volume.
    """
    _validate(df, ["date", "code", "name", vwap_col, close_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    df["_rv96"] = df.groupby("date")[vwap_col].transform(cs_rank)
    df["_rvol96"] = df.groupby("date")[volume_col].transform(cs_rank)

    def _per_stock(g):
        p1 = ts_rank(decay_linear(correlation(g["_rv96"], g["_rvol96"], 3), 4), 8)
        a60 = adv(g[volume_col], 60)
        c = correlation(ts_rank(g[close_col], 7), ts_rank(a60, 4), 4)
        p2 = ts_rank(decay_linear(ts_argmax(c, 12), 14), 13)
        return pd.Series(np.maximum(p1.values, p2.values), index=g.index)

    df["alpha096"] = (
        df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
        .reset_index(level=0, drop=True)
    ) * -1
    return _finish(df, idx, "alpha096", append, ["_rv96", "_rvol96"])


def add_alpha097(
    df: pd.DataFrame,
    low_col: str = "low",
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    neut_price_col: str = "neut_price97",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #097:
        rank(decay_linear(delta(IndNeutralize(low*0.721001+vwap*(1-0.721001)), 3), 8)) +
        ts_rank(decay_linear(ts_rank(correlation(ts_rank(low,7), ts_rank(adv60,4), 6), 4), 8), 6)

    IndNeutralize must be pre-computed and passed as ``neut_price_col``.
    Required columns: date, code, name, low, vwap, volume, ``neut_price_col``.
    """
    _validate(df, ["date", "code", "name", low_col, vwap_col, volume_col, neut_price_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        p1 = decay_linear(delta(g[neut_price_col], 3), 8)
        a60 = adv(g[volume_col], 60)
        c = correlation(ts_rank(g[low_col], 7), ts_rank(a60, 4), 6)
        p2 = ts_rank(decay_linear(ts_rank(c, 4), 8), 6)
        return pd.DataFrame({"_p197": p1, "_p297": p2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_p197"] = tmp["_p197"]
    df["_p297"] = tmp["_p297"]
    df["_rp197"] = df.groupby("date")["_p197"].transform(cs_rank)
    df["alpha097"] = df["_rp197"] + df["_p297"]
    return _finish(df, idx, "alpha097", append, ["_p197", "_p297", "_rp197"])


def add_alpha098(
    df: pd.DataFrame,
    vwap_col: str = "vwap",
    open_col: str = "open",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #098:
        rank(decay_linear(correlation(vwap, sum(adv5,26), 4), 7)) -
        rank(decay_linear(ts_rank(ts_argmin(correlation(rank(open), rank(adv15), 20), 8), 6), 4))

    Required columns: date, code, name, vwap, open, volume.
    """
    _validate(df, ["date", "code", "name", vwap_col, open_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    df["_ro98"] = df.groupby("date")[open_col].transform(cs_rank)

    def _per_stock(g):
        a5 = adv(g[volume_col], 5)
        p1 = decay_linear(correlation(g[vwap_col], ts_sum(a5, 26), 4), 7)
        a15 = adv(g[volume_col], 15)
        ra15 = cs_rank(a15)
        c = correlation(g["_ro98"], ra15, 20)
        p2 = decay_linear(ts_rank(ts_argmin(c, 8), 6), 4)
        return pd.DataFrame({"_p198": p1, "_p298": p2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_p198"] = tmp["_p198"]
    df["_p298"] = tmp["_p298"]
    df["_rp198"] = df.groupby("date")["_p198"].transform(cs_rank)
    df["_rp298"] = df.groupby("date")["_p298"].transform(cs_rank)
    df["alpha098"] = df["_rp198"] - df["_rp298"]
    return _finish(df, idx, "alpha098", append, ["_ro98", "_p198", "_p298", "_rp198", "_rp298"])


def add_alpha099(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #099:
        (rank(correlation(sum((high+low)/2, 19), sum(adv60,19), 8)) <
         rank(correlation(low, volume, 6))) * -1

    Required columns: date, code, name, high, low, volume.
    """
    _validate(df, ["date", "code", "name", high_col, low_col, volume_col])
    df = _sort(df).copy()
    idx = df.index

    def _per_stock(g):
        mid = (g[high_col] + g[low_col]) / 2
        a60 = adv(g[volume_col], 60)
        c1 = correlation(ts_sum(mid, 19), ts_sum(a60, 19), 8)
        c2 = correlation(g[low_col], g[volume_col], 6)
        return pd.DataFrame({"_c199": c1, "_c299": c2}, index=g.index)

    tmp = df.groupby("code", group_keys=False).apply(_per_stock, include_groups=False)
    df["_c199"] = tmp["_c199"]
    df["_c299"] = tmp["_c299"]
    df["_rc199"] = df.groupby("date")["_c199"].transform(cs_rank)
    df["_rc299"] = df.groupby("date")["_c299"].transform(cs_rank)
    df["alpha099"] = ((df["_rc199"] < df["_rc299"]).astype(int) * -1)
    return _finish(df, idx, "alpha099", append, ["_c199", "_c299", "_rc199", "_rc299"])


def add_alpha100(
    df: pd.DataFrame,
    neut_rank_col: str = "neut_rank100",
    neut_diff_col: str = "neut_diff100",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #100:
        -1 * rank(1.5*scale(IndNeutralize(IndNeutralize(rank(((close-low-(high-close))/(high-low))*volume)))) -
                  scale(IndNeutralize(correlation(close, rank(adv20),5) - rank(ts_argmin(close,5)))))

    Both IndNeutralize terms must be pre-computed externally:
      - ``neut_rank_col``: the double-neutralized rank term
      - ``neut_diff_col``: the neutralized (corr - argmin) term

    Required columns: date, code, name, ``neut_rank_col``, ``neut_diff_col``.
    """
    _validate(df, ["date", "code", "name", neut_rank_col, neut_diff_col])
    df = _sort(df).copy()
    idx = df.index

    df["_inner100"] = (
        1.5 * df.groupby("date")[neut_rank_col].transform(scale_alpha)
        - df.groupby("date")[neut_diff_col].transform(scale_alpha)
    )
    df["alpha100"] = -1 * df.groupby("date")["_inner100"].transform(cs_rank)
    return _finish(df, idx, "alpha100", append, ["_inner100"])


def add_alpha101(
    df: pd.DataFrame,
    close_col: str = "close",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    append: bool = True,
) -> pd.DataFrame:
    """
    Alpha #101: (close - open) / (high - low + 0.001)

    Required columns: date, code, name, close, open, high, low.
    """
    _validate(df, ["date", "code", "name", close_col, open_col, high_col, low_col])
    df = _sort(df).copy()
    idx = df.index

    df["alpha101"] = (
        (df[close_col] - df[open_col]) / (df[high_col] - df[low_col] + 0.001)
    )
    return _finish(df, idx, "alpha101", append, [])
