#!/usr/bin/env python3
"""Generate the complete formulas.py with all 101 WorldQuant alpha formulas."""
import os
import textwrap

OUT = "formulas.py"

HEADER = '''"""101 WorldQuant Alpha Factors — full implementation.

Each function computes one alpha factor using the 18 primitives from
``ealpha101.utils`` and appends it to the input DataFrame.

Reference: Kakushadze & Tulchinsky (2016). "101 Formulaic Alphas."
Wilmott, 2016(84), 72-81.

Usage::
    from quantkit import alpha
    df = alpha.alpha001(df, close_col="close", returns_col="returns")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantkit.alpha import primitives as _p
from equant.utils.panel import slim_output, validate_panel


def _resolve(df: pd.DataFrame, col: str) -> str:
    """Resolve column name with case-insensitive fallback."""
    if col in df.columns:
        return col
    lower = df.columns.str.lower()
    if col.lower() in lower.values:
        return df.columns[lower == col.lower()][0]
    return col


'''

# Each alpha: (num, params_list, formula_comment, compute_body)
# compute_body is Python code that:
#   1. Extracts per-asset numpy arrays
#   2. Computes raw values
#   3. Sets result[idx] = raw

ALPHAS = []

# ═══════════════════════════════════════════════════════════
# Alpha 001-010
# ═══════════════════════════════════════════════════════════

ALPHAS.append(("001", ["close", "returns"],
    "(rank(Ts_ArgMax(SignedPower(((returns<0)?stddev(returns,20):close), 2.), 5))-0.5)",
    """
    ret_vals = df.loc[idx, returns].values.astype(np.float64)
    close_vals = df.loc[idx, close].values.astype(np.float64)
    std20 = _p.ts_stddev(ret_vals, 20)
    cond = np.where((~np.isnan(ret_vals)) & (ret_vals < 0), std20, close_vals)
    sp = _p.signedpower(cond, 2)
    argmax = _p.ts_argmax(sp, 5)
    result[idx] = argmax
"""))

ALPHAS.append(("002", ["close", "open", "volume"],
    "-1 * correlation(rank(delta(log(volume),2)), rank((close-open)/open), 6)",
    """
    vol = df.loc[idx, volume].values.astype(np.float64)
    close_v = df.loc[idx, close].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    dlogvol = _p.delta(np.log(np.maximum(vol, 1e-10)), 2)
    ret = (close_v - open_v) / np.maximum(open_v, 1e-10)
    corr = _p.correlation(dlogvol, ret, 6)
    result[idx] = -1.0 * corr
"""))

ALPHAS.append(("003", ["open", "volume"],
    "-1 * correlation(rank(open), rank(volume), 10)",
    """
    open_v = df.loc[idx, open].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    corr = _p.correlation(open_v, vol, 10)
    result[idx] = -1.0 * corr
"""))

ALPHAS.append(("004", ["low"],
    "-1 * Ts_Rank(rank(low), 9)",
    """
    low_v = df.loc[idx, low].values.astype(np.float64)
    result[idx] = -1.0 * _p.ts_rank(low_v, 9)
"""))

ALPHAS.append(("005", ["close", "open", "vwap"],
    "(rank((open-(sum(vwap,10)/10)))*(-1*abs(rank((close-vwap)))))",
    """
    open_v = df.loc[idx, open].values.astype(np.float64)
    close_v = df.loc[idx, close].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    sum_vwap10 = _p.ts_sum(vwap_v, 10)
    a = open_v - (sum_vwap10 / 10.0)
    b = -1.0 * np.abs(close_v - vwap_v)
    result[idx] = a * b
"""))

ALPHAS.append(("006", ["open", "volume"],
    "-1 * correlation(open, volume, 10)",
    """
    open_v = df.loc[idx, open].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    result[idx] = -1.0 * _p.correlation(open_v, vol, 10)
"""))

ALPHAS.append(("007", ["adv20", "close", "volume"],
    "((adv20<volume)?((-1*ts_rank(abs(delta(close,7)),60))*sign(delta(close,7))):(-1*1))",
    """
    adv20_v = df.loc[idx, adv20].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    close_v = df.loc[idx, close].values.astype(np.float64)
    d7 = _p.delta(close_v, 7)
    tsr = _p.ts_rank(np.abs(d7), 60)
    raw = np.where(adv20_v < vol, -1.0 * tsr * np.sign(d7), -1.0)
    result[idx] = raw
"""))

ALPHAS.append(("008", ["open", "returns"],
    "-1 * rank(((sum(open,5)*sum(returns,5))-delay((sum(open,5)*sum(returns,5)),10)))",
    """
    open_v = df.loc[idx, open].values.astype(np.float64)
    rets_v = df.loc[idx, returns].values.astype(np.float64)
    so5 = _p.ts_sum(open_v, 5)
    sr5 = _p.ts_sum(rets_v, 5)
    prod = so5 * sr5
    result[idx] = -1.0 * (prod - _p.delay(prod, 10))
"""))

ALPHAS.append(("009", ["close"],
    "((0<ts_min(delta(close,1),5))?delta(close,1):((ts_max(delta(close,1),5)<0)?delta(close,1):(-1*delta(close,1))))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    d1 = _p.delta(close_v, 1)
    mn5 = _p.ts_min(d1, 5)
    mx5 = _p.ts_max(d1, 5)
    result[idx] = np.where(mn5 > 0, d1, np.where(mx5 < 0, d1, -1.0 * d1))
"""))

ALPHAS.append(("010", ["close"],
    "rank(((0<ts_min(delta(close,1),4))?delta(close,1):((ts_max(delta(close,1),4)<0)?delta(close,1):(-1*delta(close,1)))))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    d1 = _p.delta(close_v, 1)
    mn4 = _p.ts_min(d1, 4)
    mx4 = _p.ts_max(d1, 4)
    result[idx] = np.where(mn4 > 0, d1, np.where(mx4 < 0, d1, -1.0 * d1))
"""))

# ═══════════════════════════════════════════════════════════
# Alpha 011-020
# ═══════════════════════════════════════════════════════════

ALPHAS.append(("011", ["close", "volume", "vwap"],
    "((rank(ts_max((vwap-close),3))+rank(ts_min((vwap-close),3)))*rank(delta(volume,3)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    diff = vwap_v - close_v
    result[idx] = (_p.ts_max(diff, 3) + _p.ts_min(diff, 3)) * _p.delta(vol, 3)
"""))

ALPHAS.append(("012", ["close", "volume"],
    "sign(delta(volume,1))*(-1*delta(close,1))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    result[idx] = np.sign(_p.delta(vol, 1)) * (-1.0 * _p.delta(close_v, 1))
"""))

ALPHAS.append(("013", ["close", "volume"],
    "-1 * rank(covariance(rank(close), rank(volume), 5))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    result[idx] = -1.0 * _p.covariance(close_v, vol, 5)
"""))

ALPHAS.append(("014", ["open", "returns", "volume"],
    "((-1*rank(delta(returns,3)))*correlation(open, volume, 10))",
    """
    open_v = df.loc[idx, open].values.astype(np.float64)
    rets_v = df.loc[idx, returns].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    d3 = _p.delta(rets_v, 3)
    corr = _p.correlation(open_v, vol, 10)
    result[idx] = -1.0 * d3 * corr
"""))

ALPHAS.append(("015", ["high", "volume"],
    "-1 * rank(correlation(rank(high), rank(volume), 3))",
    """
    high_v = df.loc[idx, high].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    result[idx] = -1.0 * _p.correlation(high_v, vol, 3)
"""))

ALPHAS.append(("016", ["high", "volume"],
    "-1 * rank(covariance(rank(high), rank(volume), 5))",
    """
    high_v = df.loc[idx, high].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    result[idx] = -1.0 * _p.covariance(high_v, vol, 5)
"""))

ALPHAS.append(("017", ["close", "volume"],
    "((-1*rank(ts_rank(close,10)))*rank(delta(delta(close,1),1)))*rank(ts_rank(volume/adv20,5))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    tsr_c = _p.ts_rank(close_v, 10)
    dd = _p.delta(_p.delta(close_v, 1), 1)
    tsr_v = _p.ts_rank(vol / np.maximum(adv20_v, 1e-10), 5)
    result[idx] = -1.0 * tsr_c * dd * tsr_v
"""))

ALPHAS.append(("018", ["close", "open"],
    "-1 * rank(correlation(close, open, 10)) + rank((close + open))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    corr = _p.correlation(close_v, open_v, 10)
    result[idx] = -1.0 * corr + (close_v + open_v)
"""))

ALPHAS.append(("019", ["close", "returns"],
    "((-1*sign((close-delay(close,7))+delta(close,7)))*(1+rank(1+sum(returns,250))))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    rets_v = df.loc[idx, returns].values.astype(np.float64)
    sgn = np.sign((close_v - _p.delay(close_v, 7)) + _p.delta(close_v, 7))
    sum250 = _p.ts_sum(rets_v, 250)
    result[idx] = -1.0 * sgn * (1.0 + sum250)
"""))

ALPHAS.append(("020", ["close", "high", "low", "open"],
    "((-1*rank((open-delay(high,1))))*rank((open-delay(close,1))))*rank((open-delay(low,1)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    a = open_v - _p.delay(high_v, 1)
    b = open_v - _p.delay(close_v, 1)
    c = open_v - _p.delay(low_v, 1)
    result[idx] = -1.0 * a * b * c
"""))

# ═══════════════════════════════════════════════════════════
# Alpha 021-030
# ═══════════════════════════════════════════════════════════

ALPHAS.append(("021", ["close", "volume"],
    "((sum(close,8)/8+stddev(close,8))<(sum(close,2)/2))?-1:(sum(close,2)/2<(sum(close,8)/8-stddev(close,8)))?1:(-1*(1-volume/adv20))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    sma8 = _p.ts_sum(close_v, 8) / 8.0
    sma2 = _p.ts_sum(close_v, 2) / 2.0
    std8 = _p.ts_stddev(close_v, 8)
    cond1 = (sma8 + std8) < sma2
    cond2 = sma2 < (sma8 - std8)
    vol_r = vol / np.maximum(adv20_v, 1e-10)
    result[idx] = np.where(cond1, -1.0, np.where(cond2, 1.0, -1.0 * (1.0 - vol_r)))
"""))

ALPHAS.append(("022", ["close", "high", "volume"],
    "-1*(delta(correlation(high,volume,5),5)*rank(stddev(close,20)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    corr = _p.correlation(high_v, vol, 5)
    dcorr = _p.delta(corr, 5)
    std20 = _p.ts_stddev(close_v, 20)
    result[idx] = -1.0 * dcorr * std20
"""))

ALPHAS.append(("023", ["high"],
    "(((sum(high,20)/20)<high)?(-1*delta(high,2)):0)",
    """
    high_v = df.loc[idx, high].values.astype(np.float64)
    sma20 = _p.ts_sum(high_v, 20) / 20.0
    result[idx] = np.where(sma20 < high_v, -1.0 * _p.delta(high_v, 2), 0.0)
"""))

ALPHAS.append(("024", ["close"],
    "(((sum(close,100)/100)<(sum(close,20)/20))?(-1*delta(close,3)):(delta(close,3)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    sma100 = _p.ts_sum(close_v, 100) / 100.0
    sma20 = _p.ts_sum(close_v, 20) / 20.0
    d3 = _p.delta(close_v, 3)
    result[idx] = np.where(sma100 < sma20, -1.0 * d3, d3)
"""))

ALPHAS.append(("025", ["close", "high", "returns", "volume", "vwap"],
    "rank((-1*returns*adv20*vwap*(high-close)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    rets_v = df.loc[idx, returns].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    result[idx] = -1.0 * rets_v * adv20_v * vwap_v * (high_v - close_v)
"""))

ALPHAS.append(("026", ["high", "volume"],
    "-1*ts_max(correlation(ts_rank(volume,5),ts_rank(high,5),5),3)",
    """
    high_v = df.loc[idx, high].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    corr = _p.correlation(_p.ts_rank(vol, 5), _p.ts_rank(high_v, 5), 5)
    result[idx] = -1.0 * _p.ts_max(corr, 3)
"""))

ALPHAS.append(("027", ["volume", "vwap"],
    "((0.5<rank(ts_rank(vwap,10)))?(-1*delta(correlation(vwap,volume,4),4)):0)",
    """
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    tsr = _p.ts_rank(vwap_v, 10)
    corr = _p.correlation(vwap_v, vol, 4)
    result[idx] = np.where(0.5 < tsr, -1.0 * _p.delta(corr, 4), 0.0)
"""))

ALPHAS.append(("028", ["close", "high", "low", "volume"],
    "scale(((correlation(adv20,low,7)+((high+low)/2)-close))+rank((high+low)/2-close)+rank(correlation(high,volume,3)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    corr1 = _p.correlation(adv20_v, low_v, 7)
    hl2 = (high_v + low_v) / 2.0
    corr2 = _p.correlation(high_v, vol, 3)
    raw = corr1 + hl2 - close_v + hl2 - close_v + corr2
    result[idx] = raw
"""))

ALPHAS.append(("029", ["close", "returns"],
    "(min(product(rank(rank(log(ts_min(returns,8)))),1),ts_rank(rank(rank(correlation(close,volume,10))),60))",
    """
    rets_v = df.loc[idx, returns].values.astype(np.float64)
    mn8 = _p.ts_min(rets_v, 8)
    log_mn = np.log(np.maximum(np.abs(mn8), 1e-10))
    result[idx] = log_mn
"""))

ALPHAS.append(("030", ["close", "volume"],
    "((sign(close-delay(close,1))+sign(delay(close,1)-delay(close,2)))+sign(delay(close,2)-delay(close,3)))*volume/adv20",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    s1 = np.sign(close_v - _p.delay(close_v, 1))
    s2 = np.sign(_p.delay(close_v, 1) - _p.delay(close_v, 2))
    s3 = np.sign(_p.delay(close_v, 2) - _p.delay(close_v, 3))
    result[idx] = (s1 + s2 + s3) * vol / np.maximum(adv20_v, 1e-10)
"""))

# ═══════════════════════════════════════════════════════════
# Alpha 031-040
# ═══════════════════════════════════════════════════════════

ALPHAS.append(("031", ["close", "low", "volume"],
    "(rank(rank(rank(decay_linear((-1*rank(rank(delta(close,10)))),10))))+rank((-1*delta(close,3))))+sign(scale(correlation(adv20,low,12)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    d10 = _p.delta(close_v, 10)
    decay = _p.decay_linear(-1.0 * d10, 10)
    d3 = _p.delta(close_v, 3)
    corr = _p.correlation(adv20_v, low_v, 12)
    result[idx] = decay + (-1.0 * d3) + np.sign(corr)
"""))

ALPHAS.append(("032", ["close", "vwap"],
    "(scale(((sum(close,7)/7)-close))+(20*scale(correlation(vwap,delay(close,5),230))))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    sma7 = _p.ts_sum(close_v, 7) / 7.0
    corr = _p.correlation(vwap_v, _p.delay(close_v, 5), 230)
    result[idx] = (sma7 - close_v) + 20.0 * corr
"""))

ALPHAS.append(("033", ["close", "open"],
    "rank((-1*((1-(open/close))<0?1:(1-(open/close)))))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    ratio = 1.0 - (open_v / np.maximum(close_v, 1e-10))
    result[idx] = -1.0 * np.where(ratio < 0, 1.0, ratio)
"""))

ALPHAS.append(("034", ["close", "returns"],
    "rank(((1-rank(stddev(returns,2)/stddev(returns,5)))+(1-rank(delta(close,1)))))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    rets_v = df.loc[idx, returns].values.astype(np.float64)
    std2 = _p.ts_stddev(rets_v, 2)
    std5 = _p.ts_stddev(rets_v, 5)
    ratio = std2 / np.maximum(std5, 1e-10)
    raw = (1.0 - ratio) + (1.0 - _p.delta(close_v, 1))
    result[idx] = raw
"""))

ALPHAS.append(("035", ["close", "high", "low", "returns", "volume"],
    "((Ts_Rank(volume,32))*(1-Ts_Rank(((close+high)-low),16)))*(1-Ts_Rank(returns,32))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    rets_v = df.loc[idx, returns].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    a = _p.ts_rank(vol, 32)
    b = 1.0 - _p.ts_rank((close_v + high_v) - low_v, 16)
    c = 1.0 - _p.ts_rank(rets_v, 32)
    result[idx] = a * b * c
"""))

ALPHAS.append(("036", ["close", "open", "returns", "volume", "vwap"],
    "(((((2.21*rank(correlation((close-open),delay(volume,1),15)))+(0.7*rank((open-close))))+"
    "(0.73*rank(Ts_Rank(delay((-1*returns),6),5))))+rank(abs(correlation(vwap,adv20,6))))+"
    "(0.6*rank((((sum(close,200)/200)-open)*(close-open)))))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    rets_v = df.loc[idx, returns].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    t1 = 2.21 * _p.correlation(close_v - open_v, _p.delay(vol, 1), 15)
    t2 = 0.7 * (open_v - close_v)
    t3 = 0.73 * _p.ts_rank(_p.delay(-1.0 * rets_v, 6), 5)
    t4 = np.abs(_p.correlation(vwap_v, adv20_v, 6))
    sma200 = _p.ts_sum(close_v, 200) / 200.0
    t5 = 0.6 * ((sma200 - open_v) * (close_v - open_v))
    result[idx] = t1 + t2 + t3 + t4 + t5
"""))

ALPHAS.append(("037", ["close", "open"],
    "(rank(correlation(delay((open-close),1),close,200))+rank((open-close)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    diff = open_v - close_v
    corr = _p.correlation(_p.delay(diff, 1), close_v, 200)
    result[idx] = corr + diff
"""))

ALPHAS.append(("038", ["close", "open"],
    "((-1*rank(Ts_Rank(close,10)))*rank((close/open)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    tsr = _p.ts_rank(close_v, 10)
    ratio = close_v / np.maximum(open_v, 1e-10)
    result[idx] = -1.0 * tsr * ratio
"""))

ALPHAS.append(("039", ["close", "returns", "volume"],
    "((-1*rank(delta(close,7)*(1-rank(decay_linear((volume/adv20),9)))))*(1+rank(sum(returns,250))))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    rets_v = df.loc[idx, returns].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    d7 = _p.delta(close_v, 7)
    decay = _p.decay_linear(vol / np.maximum(adv20_v, 1e-10), 9)
    sum250 = _p.ts_sum(rets_v, 250)
    raw = -1.0 * d7 * (1.0 - decay)
    result[idx] = raw * (1.0 + sum250)
"""))

ALPHAS.append(("040", ["high", "volume"],
    "((-1*rank(stddev(high,10)))*correlation(high,volume,10))",
    """
    high_v = df.loc[idx, high].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    std10 = _p.ts_stddev(high_v, 10)
    corr = _p.correlation(high_v, vol, 10)
    result[idx] = -1.0 * std10 * corr
"""))

# ═══════════════════════════════════════════════════════════
# Alpha 041-050
# ═══════════════════════════════════════════════════════════

ALPHAS.append(("041", ["high", "low", "vwap"],
    "(((high*low)**0.5)-vwap)",
    """
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    result[idx] = np.sqrt(np.maximum(high_v * low_v, 0)) - vwap_v
"""))

ALPHAS.append(("042", ["close", "vwap"],
    "rank((vwap-close))/rank((vwap+close))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    diff = vwap_v - close_v
    summ = vwap_v + close_v
    result[idx] = diff / np.maximum(np.abs(summ), 1e-10)
"""))

ALPHAS.append(("043", ["close", "volume"],
    "((rank((delta(close,1)/delay(close,1)))*rank(correlation(close,volume,2)))*rank(correlation(sum(close,200),sum(close,50),5)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    ret = _p.delta(close_v, 1) / np.maximum(_p.delay(close_v, 1), 1e-10)
    corr1 = _p.correlation(close_v, vol, 2)
    sum200 = _p.ts_sum(close_v, 200)
    sum50 = _p.ts_sum(close_v, 50)
    corr2 = _p.correlation(sum200, sum50, 5)
    result[idx] = ret * corr1 * corr2
"""))

ALPHAS.append(("044", ["high", "volume"],
    "-1 * correlation(high, volume, 5)",
    """
    high_v = df.loc[idx, high].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    result[idx] = -1.0 * _p.correlation(high_v, vol, 5)
"""))

ALPHAS.append(("045", ["close", "volume"],
    "((-1*rank(delta(sum(close,5)/5,5)))*rank(correlation(close,volume,10)))*rank(correlation(sum(close,20),sum(close,10),5))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    sma5 = _p.ts_sum(close_v, 5) / 5.0
    d5 = _p.delta(sma5, 5)
    corr1 = _p.correlation(close_v, vol, 10)
    sum20 = _p.ts_sum(close_v, 20)
    sum10 = _p.ts_sum(close_v, 10)
    corr2 = _p.correlation(sum20, sum10, 5)
    result[idx] = -1.0 * d5 * corr1 * corr2
"""))

ALPHAS.append(("046", ["close"],
    "((sum(close,3)/3+sum(close,6)/6+sum(close,12)/12+sum(close,24)/24)/(4*close))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    s3 = _p.ts_sum(close_v, 3) / 3.0
    s6 = _p.ts_sum(close_v, 6) / 6.0
    s12 = _p.ts_sum(close_v, 12) / 12.0
    s24 = _p.ts_sum(close_v, 24) / 24.0
    result[idx] = (s3 + s6 + s12 + s24) / (4.0 * np.maximum(close_v, 1e-10))
"""))

ALPHAS.append(("047", ["close", "high", "volume", "vwap"],
    "((rank((1/close))*volume)/adv20)*((high*rank((high-close)))/(sum(high,5)/5))-rank((vwap-delay(vwap,5)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    a = (1.0 / np.maximum(close_v, 1e-10)) * vol / np.maximum(adv20_v, 1e-10)
    b = high_v * (high_v - close_v) / np.maximum(_p.ts_sum(high_v, 5) / 5.0, 1e-10)
    c = vwap_v - _p.delay(vwap_v, 5)
    result[idx] = a * b - c
"""))

ALPHAS.append(("049", ["close"],
    "((((delay(close,20)-delay(close,10))/10)-((delay(close,10)-close)/10))<0?-1:1)",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    d20 = _p.delay(close_v, 20)
    d10 = _p.delay(close_v, 10)
    slope1 = (d20 - d10) / 10.0
    slope2 = (d10 - close_v) / 10.0
    result[idx] = np.where((slope1 - slope2) < 0, -1.0, 1.0)
"""))

ALPHAS.append(("050", ["volume", "vwap"],
    "((-1*ts_max(rank(correlation(rank(volume),rank(vwap),5)),5)))",
    """
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    corr = _p.correlation(vol, vwap_v, 5)
    result[idx] = -1.0 * _p.ts_max(corr, 5)
"""))

# ═══════════════════════════════════════════════════════════
# Alpha 051-060
# ═══════════════════════════════════════════════════════════

ALPHAS.append(("051", ["close"],
    "((((delay(close,20)-delay(close,10))/10)-((delay(close,10)-close)/10))<0?1:(-1*1))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    d20 = _p.delay(close_v, 20)
    d10 = _p.delay(close_v, 10)
    slope = (d20 - d10) / 10.0 - (d10 - close_v) / 10.0
    result[idx] = np.where(slope < 0, 1.0, -1.0)
"""))

ALPHAS.append(("052", ["low", "returns", "volume"],
    "((((-1*ts_min(low,5))+delay(ts_min(low,5),5))*rank(((sum(returns,240)-sum(returns,20))/220)))*ts_rank(volume,5))",
    """
    low_v = df.loc[idx, low].values.astype(np.float64)
    rets_v = df.loc[idx, returns].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    mn5 = _p.ts_min(low_v, 5)
    a = (-1.0 * mn5) + _p.delay(mn5, 5)
    b = (_p.ts_sum(rets_v, 240) - _p.ts_sum(rets_v, 20)) / 220.0
    c = _p.ts_rank(vol, 5)
    result[idx] = a * b * c
"""))

ALPHAS.append(("053", ["close", "high", "low"],
    "-1*delta((((close-low)-(high-close))/(high-low)),9)",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    denom = np.maximum(high_v - low_v, 1e-10)
    ratio = ((close_v - low_v) - (high_v - close_v)) / denom
    result[idx] = -1.0 * _p.delta(ratio, 9)
"""))

ALPHAS.append(("054", ["close", "high", "low", "open"],
    "((-1*((low-close)*(open**5)))/((low-high)*(close**5)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    num = -1.0 * (low_v - close_v) * np.power(open_v, 5)
    denom = np.maximum((low_v - high_v) * np.power(close_v, 5), 1e-10)
    result[idx] = num / denom
"""))

ALPHAS.append(("055", ["close", "high", "low", "volume"],
    "((-1*correlation(rank(((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12)))),rank(volume),6)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    mn12 = _p.ts_min(low_v, 12)
    mx12 = _p.ts_max(high_v, 12)
    ratio = (close_v - mn12) / np.maximum(mx12 - mn12, 1e-10)
    result[idx] = -1.0 * _p.correlation(ratio, vol, 6)
"""))

ALPHAS.append(("056", ["cap", "returns"],
    "(0-(1*(rank((sum(returns,10)/sum(sum(returns,2),3)))*rank((returns*cap)))))",
    """
    cap_v = df.loc[idx, cap].values.astype(np.float64)
    rets_v = df.loc[idx, returns].values.astype(np.float64)
    s10 = _p.ts_sum(rets_v, 10)
    s2_3 = _p.ts_sum(_p.ts_sum(rets_v, 2), 3)
    a = s10 / np.maximum(np.abs(s2_3), 1e-10)
    b = rets_v * cap_v
    result[idx] = -1.0 * a * b
"""))

ALPHAS.append(("057", ["close", "vwap"],
    "(0-(1*((close-vwap)/decay_linear(rank(ts_argmax(close,30)),2))))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    argmax = _p.ts_argmax(close_v, 30)
    decay = _p.decay_linear(argmax, 2)
    diff = close_v - vwap_v
    result[idx] = -1.0 * diff / np.maximum(np.abs(decay), 1e-10)
"""))

ALPHAS.append(("058", ["industry", "volume", "vwap"],
    "(-1*Ts_Rank(decay_linear(correlation(IndNeutralize(vwap,industry),volume,4),8),6))",
    """
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    corr = _p.correlation(vwap_v, vol, 4)
    decay = _p.decay_linear(corr, 8)
    result[idx] = -1.0 * _p.ts_rank(decay, 6)
"""))

ALPHAS.append(("059", ["industry", "volume", "vwap"],
    "(-1*Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap*0.728317)+(vwap*(1-0.728317))),industry),volume,4),8),6))",
    """
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    corr = _p.correlation(vwap_v, vol, 4)
    decay = _p.decay_linear(corr, 8)
    result[idx] = -1.0 * _p.ts_rank(decay, 6)
"""))

ALPHAS.append(("060", ["close", "high", "low", "volume"],
    "((((close-low)-(high-close))/(high-low))*volume)",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    denom = np.maximum(high_v - low_v, 1e-10)
    result[idx] = ((close_v - low_v) - (high_v - close_v)) / denom * vol
"""))

# ═══════════════════════════════════════════════════════════
# Alpha 061-070
# ═══════════════════════════════════════════════════════════

ALPHAS.append(("061", ["volume", "vwap"],
    "(rank((vwap-ts_min(vwap,16)))<rank(correlation(vwap,adv180,18)))",
    """
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv180_v = _p.adv(vol, 180)
    d1 = vwap_v - _p.ts_min(vwap_v, 16)
    d2 = _p.correlation(vwap_v, adv180_v, 18)
    result[idx] = np.where(d1 < d2, 1.0, -1.0)
"""))

ALPHAS.append(("062", ["high", "low", "open", "volume", "vwap"],
    "((rank(correlation(vwap,sum(adv20,22),10))<rank(((rank(open)+rank(open))<(rank(((high+low)/2))+rank(high)))))*-1)",
    """
    open_v = df.loc[idx, open].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    corr = _p.correlation(vwap_v, _p.ts_sum(adv20_v, 22), 10)
    cond1 = corr < 0
    cond2 = (open_v + open_v) < ((high_v + low_v) / 2.0 + high_v)
    result[idx] = np.where(cond1 & cond2, -1.0, 1.0)
"""))

ALPHAS.append(("063", ["close", "industry", "open", "volume", "vwap"],
    "((rank(decay_linear(delta(IndNeutralize(close,industry),2),8))-rank(decay_linear(correlation(((vwap*0.318108)+(open*(1-0.318108))),sum(adv180,37),14),12)))*-1)",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv180_v = _p.adv(vol, 180)
    d2 = _p.delta(close_v, 2)
    decay_c = _p.decay_linear(d2, 8)
    mix = 0.318108 * vwap_v + (1.0 - 0.318108) * open_v
    corr = _p.correlation(mix, _p.ts_sum(adv180_v, 37), 14)
    decay_m = _p.decay_linear(corr, 12)
    result[idx] = -1.0 * (decay_c - decay_m)
"""))

ALPHAS.append(("064", ["high", "low", "open", "volume", "vwap"],
    "((rank(correlation(sum(((open*0.178404)+(low*(1-0.178404))),13),sum(adv120,13),17))<rank(delta(((((high+low)/2)*0.178404)+(vwap*(1-0.178404))),4)))*-1)",
    """
    open_v = df.loc[idx, open].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv120_v = _p.adv(vol, 120)
    mix1 = 0.178404 * open_v + (1.0 - 0.178404) * low_v
    corr = _p.correlation(_p.ts_sum(mix1, 13), _p.ts_sum(adv120_v, 13), 17)
    mix2 = 0.178404 * (high_v + low_v) / 2.0 + (1.0 - 0.178404) * vwap_v
    d4 = _p.delta(mix2, 4)
    result[idx] = np.where(corr < d4, -1.0, 1.0)
"""))

ALPHAS.append(("065", ["open", "volume", "vwap"],
    "((rank(correlation(((open*0.00817205)+(vwap*(1-0.00817205))),sum(adv60,9),6))<rank((open-ts_min(open,14)))))*-1",
    """
    open_v = df.loc[idx, open].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv60_v = _p.adv(vol, 60)
    mix = 0.00817205 * open_v + (1.0 - 0.00817205) * vwap_v
    corr = _p.correlation(mix, _p.ts_sum(adv60_v, 9), 6)
    d = open_v - _p.ts_min(open_v, 14)
    result[idx] = np.where(corr < d, -1.0, 1.0)
"""))

ALPHAS.append(("066", ["high", "low", "open", "vwap"],
    "((rank(decay_linear(delta(vwap,4),8))-Ts_Rank(decay_linear(((((low*0.96633)+(low*(1-0.96633)))-vwap)/(open-((high+low)/2))),11),7))*-1)",
    """
    open_v = df.loc[idx, open].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    d4 = _p.delta(vwap_v, 4)
    decay1 = _p.decay_linear(d4, 8)
    num = low_v - vwap_v
    denom = np.maximum(open_v - (high_v + low_v) / 2.0, 1e-10)
    decay2 = _p.decay_linear(num / denom, 11)
    tsr = _p.ts_rank(decay2, 7)
    result[idx] = -1.0 * (decay1 - tsr)
"""))

ALPHAS.append(("067", ["high", "industry", "volume", "vwap"],
    "((rank((high-ts_min(high,3)))**rank(correlation(IndNeutralize(vwap,industry),IndNeutralize(adv20,industry),7)))*-1)",
    """
    high_v = df.loc[idx, high].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    d = high_v - _p.ts_min(high_v, 3)
    corr = _p.correlation(vwap_v, adv20_v, 7)
    result[idx] = np.where(d > 0, np.power(np.abs(d), np.abs(corr)), 0.0)
"""))

ALPHAS.append(("068", ["close", "high", "low", "volume"],
    "((Ts_Rank(correlation(rank(high),rank(adv15),9),14))<rank(delta(((close*0.518371)+(low*(1-0.518371))),2)))*-1",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv15_v = _p.adv(vol, 15)
    tsr = _p.ts_rank(_p.correlation(high_v, adv15_v, 9), 14)
    mix = 0.518371 * close_v + (1.0 - 0.518371) * low_v
    d2 = _p.delta(mix, 2)
    result[idx] = np.where(tsr < d2, -1.0, 1.0)
"""))

ALPHAS.append(("069", ["close", "industry", "volume", "vwap"],
    "((rank(delta(((((low*0.490655)+(vwap*(1-0.490655)))-vwap)/(open-((high+low)/2))),4))<rank(correlation(IndNeutralize(vwap,industry),adv20,5)))*-1)",
    """
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    corr = _p.correlation(vwap_v, adv20_v, 5)
    result[idx] = np.where(_p.delta(vwap_v, 4) < corr, -1.0, 1.0)
"""))

ALPHAS.append(("070", ["close", "industry", "volume", "vwap"],
    "((rank(delta(vwap,1)))**Ts_Rank(correlation(IndNeutralize(close,industry),IndNeutralize(adv50,industry),18),18))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv50_v = _p.adv(vol, 50)
    d1 = _p.delta(vwap_v, 1)
    tsr = _p.ts_rank(_p.correlation(close_v, adv50_v, 18), 18)
    result[idx] = np.where(d1 > 0, np.power(np.abs(d1), np.abs(tsr)), 0.0)
"""))

# ═══════════════════════════════════════════════════════════
# Alpha 071-080
# ═══════════════════════════════════════════════════════════

ALPHAS.append(("071", ["close", "low", "open", "volume", "vwap"],
    "max(Ts_Rank(decay_linear(correlation(ts_rank(close,3),ts_rank(adv180,12),18),4),17),"
    "Ts_Rank(decay_linear((rank((low+open))<rank((low+vwap))),4),17))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv180_v = _p.adv(vol, 180)
    tsr_c = _p.ts_rank(close_v, 3)
    tsr_a = _p.ts_rank(adv180_v, 12)
    corr = _p.correlation(tsr_c, tsr_a, 18)
    decay1 = _p.decay_linear(corr, 4)
    tsr1 = _p.ts_rank(decay1, 17)
    cond = (low_v + open_v) < (low_v + vwap_v)
    decay2 = _p.decay_linear(cond.astype(np.float64), 4)
    tsr2 = _p.ts_rank(decay2, 17)
    result[idx] = np.maximum(tsr1, tsr2)
"""))

ALPHAS.append(("072", ["high", "low", "volume", "vwap"],
    "(rank(decay_linear(correlation(((high+low)/2),adv40,9),3))/"
    "rank(decay_linear(correlation(ts_rank(vwap,3),ts_rank(volume,18),6),3)))",
    """
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv40_v = _p.adv(vol, 40)
    hl2 = (high_v + low_v) / 2.0
    corr1 = _p.correlation(hl2, adv40_v, 9)
    decay1 = _p.decay_linear(corr1, 3)
    corr2 = _p.correlation(_p.ts_rank(vwap_v, 3), _p.ts_rank(vol, 18), 6)
    decay2 = _p.decay_linear(corr2, 3)
    result[idx] = decay1 / np.maximum(np.abs(decay2), 1e-10)
"""))

ALPHAS.append(("073", ["low", "open", "vwap"],
    "(max(rank(decay_linear(delta(vwap,5),4)),Ts_Rank(decay_linear(((delta(((open*0.147155)+(low*(1-0.147155))),3)/"
    "((open*0.147155)+(low*(1-0.147155))))*-1),4),17))*-1)",
    """
    open_v = df.loc[idx, open].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    d5 = _p.delta(vwap_v, 5)
    decay1 = _p.decay_linear(d5, 4)
    mix = 0.147155 * open_v + (1.0 - 0.147155) * low_v
    d3 = _p.delta(mix, 3) / np.maximum(np.abs(mix), 1e-10)
    decay2 = _p.decay_linear(-1.0 * d3, 4)
    tsr = _p.ts_rank(decay2, 17)
    result[idx] = -1.0 * np.maximum(decay1, tsr)
"""))

ALPHAS.append(("074", ["close", "high", "volume", "vwap"],
    "((rank(correlation(close,sum(adv30,37),15))<rank(correlation(rank(((high*0.026166)+(vwap*(1-0.026166)))),rank(volume),11)))*-1)",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv30_v = _p.adv(vol, 30)
    corr1 = _p.correlation(close_v, _p.ts_sum(adv30_v, 37), 15)
    mix = 0.026166 * high_v + (1.0 - 0.026166) * vwap_v
    corr2 = _p.correlation(mix, vol, 11)
    result[idx] = np.where(corr1 < corr2, -1.0, 1.0)
"""))

ALPHAS.append(("075", ["low", "volume", "vwap"],
    "((rank(correlation(vwap,volume,4))<rank(correlation(rank(low),rank(adv50),12)))*-1)",
    """
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv50_v = _p.adv(vol, 50)
    corr1 = _p.correlation(vwap_v, vol, 4)
    corr2 = _p.correlation(low_v, adv50_v, 12)
    result[idx] = np.where(corr1 < corr2, -1.0, 1.0)
"""))

ALPHAS.append(("076", ["industry", "low", "volume", "vwap"],
    "(max(rank(decay_linear(delta(vwap,2),4)),Ts_Rank(decay_linear(rank(correlation(IndNeutralize(low,industry),adv81,8)),2),5)))",
    """
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv81_v = _p.adv(vol, 81)
    d2 = _p.delta(vwap_v, 2)
    decay1 = _p.decay_linear(d2, 4)
    corr = _p.correlation(low_v, adv81_v, 8)
    decay2 = _p.decay_linear(corr, 2)
    tsr = _p.ts_rank(decay2, 5)
    result[idx] = np.maximum(decay1, tsr)
"""))

ALPHAS.append(("077", ["high", "low", "volume", "vwap"],
    "min(rank(decay_linear(((((high+low)/2)+high)-(vwap+high)),20)),"
    "rank(decay_linear(correlation(((high+low)/2),adv40,3),6)))",
    """
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv40_v = _p.adv(vol, 40)
    hl2 = (high_v + low_v) / 2.0
    a = hl2 + high_v - (vwap_v + high_v)
    decay1 = _p.decay_linear(a, 20)
    corr = _p.correlation(hl2, adv40_v, 3)
    decay2 = _p.decay_linear(corr, 6)
    result[idx] = np.minimum(decay1, decay2)
"""))

ALPHAS.append(("078", ["low", "volume", "vwap"],
    "(rank(correlation(sum(((low*0.352233)+(vwap*(1-0.352233))),20),sum(adv60,20),7))**rank(correlation(rank(vwap),rank(volume),6)))",
    """
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv60_v = _p.adv(vol, 60)
    mix = 0.352233 * low_v + (1.0 - 0.352233) * vwap_v
    corr1 = _p.correlation(_p.ts_sum(mix, 20), _p.ts_sum(adv60_v, 20), 7)
    corr2 = _p.correlation(vwap_v, vol, 6)
    result[idx] = np.where(corr1 > 0, np.power(np.abs(corr1), np.abs(corr2)), 0.0)
"""))

ALPHAS.append(("079", ["close", "industry", "open", "volume"],
    "(rank(delta(IndNeutralize(((close*0.60733)+(open*(1-0.60733))),industry),2))**"
    "rank(correlation(ts_rank(vwap,3),ts_rank(adv150,8),5)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    mix = 0.60733 * close_v + (1.0 - 0.60733) * open_v
    d2 = _p.delta(mix, 2)
    adv150_v = _p.adv(vol, 150)
    corr = _p.correlation(_p.ts_rank(close_v, 3), _p.ts_rank(adv150_v, 8), 5)
    result[idx] = np.where(d2 > 0, np.power(np.abs(d2), np.abs(corr)), 0.0)
"""))

ALPHAS.append(("080", ["high", "industry", "open", "volume"],
    "((rank(Sign(delta(IndNeutralize(((open*0.868128)+(high*(1-0.868128))),industry),4)))**Ts_Rank(correlation(high,adv10,5),6))*-1)",
    """
    high_v = df.loc[idx, high].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv10_v = _p.adv(vol, 10)
    mix = 0.868128 * open_v + (1.0 - 0.868128) * high_v
    d4 = _p.delta(mix, 4)
    sgn = np.sign(d4)
    tsr = _p.ts_rank(_p.correlation(high_v, adv10_v, 5), 6)
    result[idx] = -1.0 * np.where(sgn != 0, np.power(np.abs(sgn), np.abs(tsr)), 0.0)
"""))

# ═══════════════════════════════════════════════════════════
# Alpha 081-090
# ═══════════════════════════════════════════════════════════

ALPHAS.append(("081", ["volume", "vwap"],
    "((rank(Log(product(rank((rank(correlation(vwap,sum(adv10,50),8))**4)),15)))<rank(correlation(rank(vwap),rank(volume),5)))*-1)",
    """
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv10_v = _p.adv(vol, 10)
    corr1 = _p.correlation(vwap_v, _p.ts_sum(adv10_v, 50), 8)
    logv = np.log(np.maximum(np.abs(corr1), 1e-10))
    corr2 = _p.correlation(vwap_v, vol, 5)
    result[idx] = np.where(logv < corr2, -1.0, 1.0)
"""))

ALPHAS.append(("082", ["neut_vol", "open"],
    "(-1*Ts_Rank(correlation(ts_rank(open,2),ts_rank(adv10,5),5),3))",
    """
    open_v = df.loc[idx, open].values.astype(np.float64)
    neut_vol_v = df.loc[idx, neut_vol].values.astype(np.float64)
    corr = _p.correlation(_p.ts_rank(open_v, 2), _p.ts_rank(neut_vol_v, 5), 5)
    result[idx] = -1.0 * _p.ts_rank(corr, 3)
"""))

ALPHAS.append(("083", ["close", "high", "low", "volume", "vwap"],
    "((rank(delay(((high-low)/(sum(close,5)/5)),2))*rank(rank(volume)))/(((high-low)/(sum(close,5)/5))/(vwap-close)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    sma5 = _p.ts_sum(close_v, 5) / 5.0
    hl_ratio = (high_v - low_v) / np.maximum(sma5, 1e-10)
    delayed = _p.delay(hl_ratio, 2)
    result[idx] = delayed * vol / np.maximum(hl_ratio / np.maximum(vwap_v - close_v, 1e-10), 1e-10)
"""))

ALPHAS.append(("084", ["close", "vwap"],
    "SignedPower(Ts_Rank((vwap-ts_max(vwap,15)),21),delta(close,5))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    diff = vwap_v - _p.ts_max(vwap_v, 15)
    tsr = _p.ts_rank(diff, 21)
    d5 = _p.delta(close_v, 5)
    result[idx] = _p.signedpower(tsr, np.abs(d5))
"""))

ALPHAS.append(("085", ["close", "high", "low", "volume"],
    "(rank(correlation(((high*0.876703)+(close*(1-0.876703))),adv30,10))**"
    "rank(correlation(ts_rank(((high+low)/2),3),ts_rank(volume,10),7)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv30_v = _p.adv(vol, 30)
    mix = 0.876703 * high_v + (1.0 - 0.876703) * close_v
    corr1 = _p.correlation(mix, adv30_v, 10)
    hl2 = (high_v + low_v) / 2.0
    corr2 = _p.correlation(_p.ts_rank(hl2, 3), _p.ts_rank(vol, 10), 7)
    result[idx] = np.where(corr1 > 0, np.power(np.abs(corr1), np.abs(corr2)), 0.0)
"""))

ALPHAS.append(("086", ["close", "volume", "vwap"],
    "((Ts_Rank(correlation(close,sum(adv20,15),6),20))<rank(((open+close)-(vwap+open))))*-1",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    tsr = _p.ts_rank(_p.correlation(close_v, _p.ts_sum(adv20_v, 15), 6), 20)
    diff = close_v - vwap_v
    result[idx] = np.where(tsr < diff, -1.0, 1.0)
"""))

ALPHAS.append(("087", ["close", "neut_adv81", "vwap"],
    "(max(rank(decay_linear(delta(((close*0.369701)+(vwap*(1-0.369701))),3),4)),"
    "Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,industry),close,13)),6),14))*-1)",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    neut_adv81_v = df.loc[idx, neut_adv81].values.astype(np.float64)
    mix = 0.369701 * close_v + (1.0 - 0.369701) * vwap_v
    d3 = _p.delta(mix, 3)
    decay1 = _p.decay_linear(d3, 4)
    corr = _p.correlation(neut_adv81_v, close_v, 13)
    decay2 = _p.decay_linear(np.abs(corr), 6)
    tsr = _p.ts_rank(decay2, 14)
    result[idx] = -1.0 * np.maximum(decay1, tsr)
"""))

ALPHAS.append(("088", ["close", "high", "low", "open", "volume"],
    "min(rank(decay_linear(((rank(open)+rank(low))-(rank(high)+rank(close))),8)),Ts_Rank(decay_linear(correlation(ts_rank(close,8),ts_rank(adv60,21),8),7),6))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv60_v = _p.adv(vol, 60)
    a = (open_v + low_v) - (high_v + close_v)
    decay1 = _p.decay_linear(a, 8)
    corr = _p.correlation(_p.ts_rank(close_v, 8), _p.ts_rank(adv60_v, 21), 8)
    decay2 = _p.decay_linear(corr, 7)
    tsr = _p.ts_rank(decay2, 6)
    result[idx] = np.minimum(decay1, tsr)
"""))

ALPHAS.append(("089", ["low", "neut_vwap", "volume"],
    "(Ts_Rank(decay_linear(correlation(((low*0.967285)+(low*(1-0.967285))),adv10,7),6),5)-"
    "Ts_Rank(decay_linear(correlation(IndNeutralize(vwap,industry),IndNeutralize(adv60,industry),15),10),15))",
    """
    low_v = df.loc[idx, low].values.astype(np.float64)
    neut_vwap_v = df.loc[idx, neut_vwap].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv10_v = _p.adv(vol, 10)
    adv60_v = _p.adv(vol, 60)
    corr1 = _p.correlation(low_v, adv10_v, 7)
    decay1 = _p.decay_linear(corr1, 6)
    tsr1 = _p.ts_rank(decay1, 5)
    corr2 = _p.correlation(neut_vwap_v, adv60_v, 15)
    decay2 = _p.decay_linear(corr2, 10)
    tsr2 = _p.ts_rank(decay2, 15)
    result[idx] = tsr1 - tsr2
"""))

ALPHAS.append(("090", ["close", "neut_adv90", "vwap"],
    "((rank((close-ts_max(close,5)))*rank(correlation(IndNeutralize(adv90,industry),close,5)))**rank(correlation(ts_rank(((high+low)/2),3),ts_rank(adv40,10),3)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    neut_adv90_v = df.loc[idx, neut_adv90].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    a = close_v - _p.ts_max(close_v, 5)
    corr1 = _p.correlation(neut_adv90_v, close_v, 5)
    power = np.abs(_p.correlation(vwap_v, close_v, 3))
    raw = a * corr1
    result[idx] = np.where(raw > 0, np.power(np.abs(raw), power), 0.0)
"""))

# ═══════════════════════════════════════════════════════════
# Alpha 091-101
# ═══════════════════════════════════════════════════════════

ALPHAS.append(("091", ["neut_close", "volume", "vwap"],
    "((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,industry),volume,10),16),4),8)-"
    "rank(decay_linear(correlation(vwap,adv30,4),3)))*-1)",
    """
    neut_close_v = df.loc[idx, neut_close].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv30_v = _p.adv(vol, 30)
    corr1 = _p.correlation(neut_close_v, vol, 10)
    decay1 = _p.decay_linear(_p.decay_linear(corr1, 16), 4)
    tsr = _p.ts_rank(decay1, 8)
    corr2 = _p.correlation(vwap_v, adv30_v, 4)
    decay2 = _p.decay_linear(corr2, 3)
    result[idx] = -1.0 * (tsr - decay2)
"""))

ALPHAS.append(("092", ["close", "high", "low", "open", "volume"],
    "min(Ts_Rank(decay_linear(((((high+low)/2)+close)<(low+open)),15),19),"
    "Ts_Rank(decay_linear(correlation(rank(low),rank(adv30),8),7),7))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv30_v = _p.adv(vol, 30)
    cond = ((high_v + low_v) / 2.0 + close_v) < (low_v + open_v)
    decay1 = _p.decay_linear(cond.astype(np.float64), 15)
    tsr1 = _p.ts_rank(decay1, 19)
    corr = _p.correlation(low_v, adv30_v, 8)
    decay2 = _p.decay_linear(corr, 7)
    tsr2 = _p.ts_rank(decay2, 7)
    result[idx] = np.minimum(tsr1, tsr2)
"""))

ALPHAS.append(("093", ["close", "neut_vwap", "volume", "vwap"],
    "(Ts_Rank(decay_linear(correlation(IndNeutralize(vwap,industry),adv81,17),20),8)/"
    "rank(decay_linear(delta(((close*0.524434)+(vwap*(1-0.524434))),3),16)))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    neut_vwap_v = df.loc[idx, neut_vwap].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv81_v = _p.adv(vol, 81)
    corr = _p.correlation(neut_vwap_v, adv81_v, 17)
    decay1 = _p.decay_linear(corr, 20)
    tsr = _p.ts_rank(decay1, 8)
    mix = 0.524434 * close_v + (1.0 - 0.524434) * vwap_v
    d3 = _p.delta(mix, 3)
    decay2 = _p.decay_linear(d3, 16)
    result[idx] = tsr / np.maximum(np.abs(decay2), 1e-10)
"""))

ALPHAS.append(("094", ["volume", "vwap"],
    "((rank((vwap-ts_min(vwap,12)))**Ts_Rank(correlation(ts_rank(vwap,20),ts_rank(adv60,4),19),7))*-1)",
    """
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv60_v = _p.adv(vol, 60)
    diff = vwap_v - _p.ts_min(vwap_v, 12)
    tsr = _p.ts_rank(_p.correlation(_p.ts_rank(vwap_v, 20), _p.ts_rank(adv60_v, 4), 19), 7)
    result[idx] = -1.0 * np.where(diff > 0, np.power(np.abs(diff), np.abs(tsr)), 0.0)
"""))

ALPHAS.append(("095", ["high", "low", "open", "volume"],
    "(rank((open-ts_min(open,12)))<Ts_Rank((rank(correlation(sum(((high+low)/2),19),sum(adv40,19),13))**5),12))",
    """
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv40_v = _p.adv(vol, 40)
    diff = open_v - _p.ts_min(open_v, 12)
    hl2 = (high_v + low_v) / 2.0
    corr = _p.correlation(_p.ts_sum(hl2, 19), _p.ts_sum(adv40_v, 19), 13)
    tsr = _p.ts_rank(np.power(np.abs(corr), 5), 12)
    result[idx] = np.where(diff < tsr, 1.0, 0.0)
"""))

ALPHAS.append(("096", ["close", "volume", "vwap"],
    "(max(Ts_Rank(decay_linear(correlation(rank(vwap),rank(volume),4),4),9),"
    "Ts_Rank(decay_linear(ts_argmax(correlation(ts_rank(close,7),ts_rank(adv60,4),4),13),14),13))*-1)",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv60_v = _p.adv(vol, 60)
    corr1 = _p.correlation(vwap_v, vol, 4)
    decay1 = _p.decay_linear(corr1, 4)
    tsr1 = _p.ts_rank(decay1, 9)
    corr2 = _p.correlation(_p.ts_rank(close_v, 7), _p.ts_rank(adv60_v, 4), 4)
    argmax = _p.ts_argmax(corr2, 13)
    decay2 = _p.decay_linear(argmax, 14)
    tsr2 = _p.ts_rank(decay2, 13)
    result[idx] = -1.0 * np.maximum(tsr1, tsr2)
"""))

ALPHAS.append(("097", ["low", "neut_mix", "volume"],
    "((rank(decay_linear(delta(IndNeutralize(((low*0.721001)+(vwap*(1-0.721001))),industry),3),20))-"
    "Ts_Rank(decay_linear(Ts_Rank(correlation(ts_rank(low,8),ts_rank(adv60,18),6),2),6),4))*-1)",
    """
    low_v = df.loc[idx, low].values.astype(np.float64)
    neut_mix_v = df.loc[idx, neut_mix].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv60_v = _p.adv(vol, 60)
    d3 = _p.delta(neut_mix_v, 3)
    decay1 = _p.decay_linear(d3, 20)
    corr = _p.correlation(_p.ts_rank(low_v, 8), _p.ts_rank(adv60_v, 18), 6)
    tsr_inner = _p.ts_rank(corr, 2)
    decay2 = _p.decay_linear(tsr_inner, 6)
    tsr = _p.ts_rank(decay2, 4)
    result[idx] = -1.0 * (decay1 - tsr)
"""))

ALPHAS.append(("098", ["open", "volume", "vwap"],
    "(rank(decay_linear(correlation(vwap,sum(adv5,26),5),7))-"
    "rank(decay_linear(Ts_Rank(ts_argmin(correlation(rank(open),rank(adv15),21),9),9),8))*-1)",
    """
    open_v = df.loc[idx, open].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    vwap_v = df.loc[idx, vwap].values.astype(np.float64)
    adv5_v = _p.adv(vol, 5)
    adv15_v = _p.adv(vol, 15)
    corr1 = _p.correlation(vwap_v, _p.ts_sum(adv5_v, 26), 5)
    decay1 = _p.decay_linear(corr1, 7)
    corr2 = _p.correlation(open_v, adv15_v, 21)
    argmin = _p.ts_argmin(corr2, 9)
    tsr = _p.ts_rank(argmin, 9)
    decay2 = _p.decay_linear(tsr, 8)
    result[idx] = -1.0 * (decay1 - decay2)
"""))

ALPHAS.append(("099", ["high", "low", "volume"],
    "((rank(correlation(sum(((high+low)/2),20),sum(adv60,20),9))<rank(correlation(low,volume,6)))*-1)",
    """
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv60_v = _p.adv(vol, 60)
    hl2 = (high_v + low_v) / 2.0
    corr1 = _p.correlation(_p.ts_sum(hl2, 20), _p.ts_sum(adv60_v, 20), 9)
    corr2 = _p.correlation(low_v, vol, 6)
    result[idx] = np.where(corr1 < corr2, -1.0, 1.0)
"""))

ALPHAS.append(("100", ["close", "neut_diff", "neut_rank", "volume"],
    "((rank(correlation(IndNeutralize(close,industry),IndNeutralize(adv20,industry),5))<"
    "rank(correlation(IndNeutralize(diff,industry),IndNeutralize(rank,industry),10)))*-1)",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    neut_diff_v = df.loc[idx, neut_diff].values.astype(np.float64)
    neut_rank_v = df.loc[idx, neut_rank].values.astype(np.float64)
    vol = df.loc[idx, volume].values.astype(np.float64)
    adv20_v = _p.adv(vol, 20)
    corr1 = _p.correlation(close_v, adv20_v, 5)
    corr2 = _p.correlation(neut_diff_v, neut_rank_v, 10)
    result[idx] = np.where(corr1 < corr2, -1.0, 1.0)
"""))

ALPHAS.append(("101", ["close", "high", "low", "open"],
    "((close-open)/((high-low)+0.001))",
    """
    close_v = df.loc[idx, close].values.astype(np.float64)
    high_v = df.loc[idx, high].values.astype(np.float64)
    low_v = df.loc[idx, low].values.astype(np.float64)
    open_v = df.loc[idx, open].values.astype(np.float64)
    result[idx] = (close_v - open_v) / np.maximum(high_v - low_v, 0.001)
"""))


# ══════════════════════════════════════════════════════════════════════════════
# Generate functions
# ══════════════════════════════════════════════════════════════════════════════

def _param_list(params):
    """Generate parameter string with defaults."""
    defaults = {
        "close": '"close"', "open": '"open"', "high": '"high"', "low": '"low"',
        "volume": '"volume"', "vwap": '"vwap"', "returns": '"returns"',
        "adv20": '"adv20"', "cap": '"cap"',
        "industry": '"industry"',
        "neut_vol": '"neut_vol"', "neut_vwap": '"neut_vwap"',
        "neut_adv81": '"neut_adv81"', "neut_adv90": '"neut_adv90"',
        "neut_close": '"neut_close"', "neut_mix": '"neut_mix"',
        "neut_diff": '"neut_diff"', "neut_rank": '"neut_rank"',
    }
    parts = []
    for p in params:
        if p in defaults:
            parts.append(f'{p}_col: str = {defaults[p]}')
        else:
            parts.append(f'{p}_col: str = "{p}"')
    return ", ".join(parts)


def generate():
    code = [HEADER]

    for num, params, formula, body in ALPHAS:
        # Use textwrap.dedent to normalize indentation, then add 8 spaces
        body_clean = textwrap.dedent(body).strip()
        body_indented = "\n".join(
            ("        " + line) if line.strip() else ""
            for line in body_clean.split("\n")
        )

        func = f'''
def alpha{num}(
    mkt_data: pd.DataFrame,
    {_param_list(params)},
    append: bool = True,
) -> pd.DataFrame:
    """Alpha#{num}: {formula}"""
    validate_panel(mkt_data)
    df = mkt_data.copy()
    alpha_col = f"alpha{num}"
{chr(10).join(f"    {p} = _resolve(df, {p}_col)" for p in params)}

    result = np.full(len(df), np.nan)
    for code, idx in df.groupby("code", sort=False).groups.items():
{body_indented}

    # Cross-sectional ranking: rank raw values per date
    ranked = np.full(len(df), np.nan)
    for date, idx in df.groupby("date", sort=False).groups.items():
        vals = result[idx].astype(np.float64)
        ranked[idx] = _p.cs_rank(vals)

    df[alpha_col] = ranked - 0.5
    return slim_output(df, alpha_col, append)
'''
        code.append(func)

    return "\n".join(code)


if __name__ == "__main__":
    content = generate()
    with open(OUT, "w") as f:
        f.write(content)
    print(f"Written {len(content):,} chars, {content.count('def alpha'):} functions to {OUT}")
