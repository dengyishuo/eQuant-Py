"""Signal generation — eBacktestCraft add_signal equivalent."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from equant.utils.panel import validate_panel


def signal(
    df: pd.DataFrame,
    indicator_cols: Optional[Sequence[str]] = None,
    signal_type: str = "threshold",
    # -- threshold / between --
    threshold: float = 0.0,
    compare_op: str = ">",
    between_lower: Optional[float] = None,
    between_upper: Optional[float] = None,
    # -- crossover --
    cross_upper: Optional[Union[float, str]] = None,
    cross_lower: Optional[Union[float, str]] = None,
    # -- multi_condition / and / or / vote --
    signal_cols: Optional[Sequence[str]] = None,
    logic_op: str = "&",
    min_votes: Optional[int] = None,
    # -- rank / quantile / percentile / zscore / score --
    top_n: int = 3,
    ascending: bool = False,
    q: float = 0.2,
    pct: float = 0.2,
    select: str = "top",
    z_threshold: float = 1.0,
    weights: Optional[Sequence[float]] = None,
    normalize: bool = True,
    score_col: str = "composite_score",
    # -- rolling / mean_reversion --
    window: int = 20,
    n_sd: float = 2.0,
    center_type: str = "mean",
    direction: str = "above",
    k: float = 2.0,
    mode: str = "long_only",
    # -- consecutive --
    condition_col: Optional[str] = None,
    n_consecutive: int = 2,
    # -- ma_cross --
    close_col: str = "close",
    fast_n: int = 5,
    slow_n: int = 20,
    fast_col: Optional[str] = None,
    slow_col: Optional[str] = None,
    # -- breakout --
    n: int = 20,
    # -- regime --
    ma_col: Optional[str] = None,
    # -- vol_regime --
    vol_n: int = 20,
    vol_method: str = "threshold",
    vol_threshold: float = 0.20,
    hist_n: int = 252,
    pct_threshold: float = 0.5,
    # -- td_setup --
    setup_bull_col: str = "td_setup_bull",
    setup_bear_col: str = "td_setup_bear",
    td_window: int = 0,
    # -- earnings / index_rebalance --
    earnings_dates: Optional[pd.DataFrame] = None,
    pre_window: int = 0,
    post_window: int = 5,
    rebalance_events: Optional[pd.DataFrame] = None,
    direction_filter: str = "both",
    # -- macro --
    macro_data: Optional[pd.DataFrame] = None,
    macro_method: str = "threshold",
    macro_col: str = "value",
    ma_n: int = 12,
    change_n: int = 1,
    # -- window (extend) --
    signal_col: Optional[str] = None,
    carry_value: Optional[int] = 1,
    extend_window: int = 5,
    # -- constant --
    constant_value: int = 1,
    # -- output --
    signal_name: Optional[str] = None,
) -> pd.DataFrame:
    """Generate 0/1 (or -1/0/1) trading signals.

    Parameters
    ----------
    df : DataFrame
        Long-format panel with at minimum 'date' and 'code' columns.
    signal_type : str
        One of: threshold, crossover, multi_condition, between, constant,
        rank, quantile, percentile, zscore, rolling, consecutive, ma_cross,
        breakout, mean_reversion, regime, score, and, or, vote,
        td_setup, earnings, index_rebalance, vol_regime, macro, window.
    """
    validate_panel(df)
    result = df.copy()

    # ── dispatch ──────────────────────────────────────────────────────────────
    if signal_type == "constant":
        col = signal_name or f"signal_constant_{constant_value}"
        result[col] = constant_value
        _log(col, result[col])
        return result

    # ── threshold ─────────────────────────────────────────────────────────────
    if signal_type == "threshold":
        _require(indicator_cols, "threshold")
        col = signal_name or (
            f"signal_{indicator_cols[0]}"
            f"_{_op_slug(compare_op)}_{threshold}"
        )
        ops = {
            ">": np.greater, "<": np.less,
            ">=": np.greater_equal, "<=": np.less_equal,
            "==": np.equal, "!=": np.not_equal,
        }
        fn = ops[compare_op]
        sig = np.ones(len(result), dtype=bool)
        for c in indicator_cols:
            v = result[c].values.astype(float)
            sig &= fn(v, threshold)
        result[col] = sig.astype(int)
        _log(col, result[col])
        return result

    # ── between ───────────────────────────────────────────────────────────────
    if signal_type == "between":
        _require(indicator_cols, "between")
        col = signal_name or (
            f"signal_{indicator_cols[0]}_between_{between_lower}_{between_upper}"
        )
        v = result[indicator_cols[0]].values.astype(float)
        cond = (v >= between_lower) & (v <= between_upper)
        result[col] = np.where(np.isnan(v), 0, cond.astype(int))
        _log(col, result[col])
        return result

    # ── crossover ─────────────────────────────────────────────────────────────
    if signal_type == "crossover":
        _require(indicator_cols, "crossover")
        c = indicator_cols[0]
        v = result[c].fillna(0).values
        if cross_lower is not None:
            lower = _resolve_band(result, cross_lower)
            cross = (v < lower) & (np.roll(v, 1) > np.roll(lower, 1))
            suffix = "cross_down"
        else:
            upper = _resolve_band(result, cross_upper)
            cross = (v > upper) & (np.roll(v, 1) < np.roll(upper, 1))
            suffix = "cross_up"
        cross[0] = False
        col = signal_name or f"signal_{c}_{suffix}"
        result[col] = cross.astype(int)
        _log(col, result[col])
        return result

    # ── multi_condition ───────────────────────────────────────────────────────
    if signal_type == "multi_condition":
        _require(indicator_cols, "multi_condition")
        logic = "and" if logic_op == "&" else "or"
        col = signal_name or f"signal_{'_'.join(indicator_cols)}_{logic}"
        sig = None
        for c in indicator_cols:
            cond = result[c].fillna(0).values > 0
            sig = cond if sig is None else (sig & cond if logic_op == "&" else sig | cond)
        result[col] = (sig.astype(int) if sig is not None else 0)
        _log(col, result[col])
        return result

    # ── and ───────────────────────────────────────────────────────────────────
    if signal_type == "and":
        cols = _signal_cols(signal_cols, indicator_cols, "and")
        col = signal_name or f"signal_and_{'_'.join(cols)}"
        sig = np.ones(len(result), dtype=bool)
        for c in cols:
            v = result[c].fillna(0).values
            sig &= v.astype(bool)
        result[col] = sig.astype(int)
        _log(col, result[col])
        return result

    # ── or ────────────────────────────────────────────────────────────────────
    if signal_type == "or":
        cols = _signal_cols(signal_cols, indicator_cols, "or")
        col = signal_name or f"signal_or_{'_'.join(cols)}"
        sig = np.zeros(len(result), dtype=bool)
        for c in cols:
            v = result[c].fillna(0).values
            sig |= v.astype(bool)
        result[col] = sig.astype(int)
        _log(col, result[col])
        return result

    # ── vote ──────────────────────────────────────────────────────────────────
    if signal_type == "vote":
        cols = _signal_cols(signal_cols, indicator_cols, "vote")
        mv = min_votes if min_votes is not None else (len(cols) // 2 + 1)
        col = signal_name or f"signal_vote_{mv}of{len(cols)}"
        mat = np.column_stack([result[c].fillna(0).values.astype(int) for c in cols])
        result[col] = (mat.sum(axis=1) >= mv).astype(int)
        _log(col, result[col])
        return result

    # ── rank (cross-sectional) ────────────────────────────────────────────────
    if signal_type == "rank":
        _require(indicator_cols, "rank")
        c = indicator_cols[0]
        col = signal_name or f"signal_rank_{c}_top{top_n}"
        asc = ascending
        result[col] = (
            result.groupby("date")[c]
            .rank(ascending=asc, method="first")
            .transform(lambda r: (r <= top_n).astype(int))
        )
        _log(col, result[col])
        return result

    # ── quantile (cross-sectional) ────────────────────────────────────────────
    if signal_type == "quantile":
        _require(indicator_cols, "quantile")
        c = indicator_cols[0]
        s = select
        col = signal_name or f"signal_quantile_{c}_{s}_{q}"

        def _qflag(grp):
            v = grp.values.astype(float)
            if s == "top":
                cutoff = np.nanquantile(v, 1 - q)
                return pd.Series((v >= cutoff).astype(int), index=grp.index)
            else:
                cutoff = np.nanquantile(v, q)
                return pd.Series((v <= cutoff).astype(int), index=grp.index)

        result[col] = result.groupby("date")[c].transform(_qflag)
        _log(col, result[col])
        return result

    # ── percentile (alias of quantile with pct param) ─────────────────────────
    if signal_type == "percentile":
        _require(indicator_cols, "percentile")
        c = indicator_cols[0]
        s = select
        col = signal_name or f"signal_pct_{c}_{s}_{pct}"

        def _pctflag(grp):
            v = grp.values.astype(float)
            if s == "top":
                cutoff = np.nanquantile(v, 1 - pct)
                return pd.Series((v >= cutoff).astype(int), index=grp.index)
            else:
                cutoff = np.nanquantile(v, pct)
                return pd.Series((v <= cutoff).astype(int), index=grp.index)

        result[col] = result.groupby("date")[c].transform(_pctflag)
        _log(col, result[col])
        return result

    # ── zscore (cross-sectional) ──────────────────────────────────────────────
    if signal_type == "zscore":
        _require(indicator_cols, "zscore")
        c = indicator_cols[0]
        col = signal_name or f"signal_zscore_{c}_z{z_threshold}"

        def _zflag(grp):
            v = grp.values.astype(float)
            mu = np.nanmean(v)
            sd = np.nanstd(v, ddof=1)
            z = (v - mu) / sd if sd > 0 else np.zeros_like(v)
            ops = {
                ">": np.greater, "<": np.less,
                ">=": np.greater_equal, "<=": np.less_equal,
            }
            fn = ops.get(compare_op, np.greater)
            return pd.Series(fn(z, z_threshold).astype(int), index=grp.index)

        result[col] = result.groupby("date")[c].transform(_zflag)
        _log(col, result[col])
        return result

    # ── rolling (time-series, per code) ──────────────────────────────────────
    if signal_type == "rolling":
        _require(indicator_cols, "rolling")
        c = indicator_cols[0]
        col = signal_name or f"signal_rolling_{c}_{direction}"
        result = result.sort_values(["code", "date"])

        def _rollflag(grp):
            v = grp[c].values.astype(float)
            nr = len(v)
            sig = np.zeros(nr, dtype=int)
            for i in range(window, nr):
                w = v[i - window: i]
                if center_type == "mean":
                    center_val = np.nanmean(w)
                else:
                    center_val = np.nanmedian(w)
                sd_val = np.nanstd(w, ddof=1)
                upper = center_val + n_sd * sd_val
                lower = center_val - n_sd * sd_val
                vi = v[i]
                if np.isnan(vi):
                    continue
                if direction == "above":
                    sig[i] = int(vi > upper)
                elif direction == "below":
                    sig[i] = int(vi < lower)
                elif direction == "cross_above":
                    prev = v[i - 1] if i > 0 else np.nan
                    sig[i] = int((not np.isnan(prev)) and prev <= upper and vi > upper)
                elif direction == "cross_below":
                    prev = v[i - 1] if i > 0 else np.nan
                    sig[i] = int((not np.isnan(prev)) and prev >= lower and vi < lower)
            return pd.Series(sig, index=grp.index)

        result[col] = result.groupby("code", group_keys=False).apply(_rollflag)
        result = result.sort_values(["date", "code"])
        _log(col, result[col])
        return result

    # ── consecutive ───────────────────────────────────────────────────────────
    if signal_type == "consecutive":
        cond_c = condition_col or (indicator_cols[0] if indicator_cols else None)
        if cond_c is None:
            raise ValueError("consecutive requires condition_col or indicator_cols")
        col = signal_name or f"signal_{cond_c}_consecutive{n_consecutive}"
        result = result.sort_values(["code", "date"])

        def _conseq(grp):
            v = (grp[cond_c].fillna(0).values > 0).astype(int)
            streak = np.zeros(len(v), dtype=int)
            for i in range(len(v)):
                if v[i]:
                    streak[i] = streak[i - 1] + 1 if i > 0 else 1
            return pd.Series((streak >= n_consecutive).astype(int), index=grp.index)

        result[col] = result.groupby("code", group_keys=False).apply(_conseq)
        result = result.sort_values(["date", "code"])
        _log(col, result[col])
        return result

    # ── ma_cross ──────────────────────────────────────────────────────────────
    if signal_type == "ma_cross":
        ma_mode = mode if mode in ("golden", "death", "both") else "golden"
        col = signal_name or f"signal_ma{fast_n}_cross_ma{slow_n}"
        result = result.sort_values(["code", "date"])

        def _macross(grp):
            px = grp[fast_col if fast_col else close_col].values.astype(float)
            fast = _rolling_ma(px, fast_n) if fast_col is None else grp[fast_col].values.astype(float)
            slow_px = grp[close_col].values.astype(float)
            slow = _rolling_ma(slow_px, slow_n) if slow_col is None else grp[slow_col].values.astype(float)
            nr = len(px)
            sig = np.zeros(nr, dtype=int)
            for i in range(1, nr):
                if any(np.isnan([fast[i], slow[i], fast[i-1], slow[i-1]])):
                    continue
                golden = fast[i] > slow[i] and fast[i-1] <= slow[i-1]
                death = fast[i] < slow[i] and fast[i-1] >= slow[i-1]
                if ma_mode == "golden":
                    sig[i] = 1 if golden else 0
                elif ma_mode == "death":
                    sig[i] = 1 if death else 0
                else:
                    sig[i] = 1 if golden else (-1 if death else 0)
            return pd.Series(sig, index=grp.index)

        result[col] = result.groupby("code", group_keys=False).apply(_macross)
        result = result.sort_values(["date", "code"])
        _log(col, result[col])
        return result

    # ── breakout ──────────────────────────────────────────────────────────────
    if signal_type == "breakout":
        bo_mode = mode if mode in ("up", "down", "both") else "up"
        col = signal_name or f"signal_breakout_{n}_{bo_mode}"
        result = result.sort_values(["code", "date"])

        def _breakout(grp):
            px = grp[close_col].values.astype(float)
            nr = len(px)
            sig = np.zeros(nr, dtype=int)
            for i in range(n + 1, nr):
                w = px[i - n: i]
                hi = np.nanmax(w)
                lo = np.nanmin(w)
                vi = px[i]
                if np.isnan(vi):
                    continue
                if bo_mode == "up":
                    sig[i] = 1 if vi > hi else 0
                elif bo_mode == "down":
                    sig[i] = 1 if vi < lo else 0
                else:
                    sig[i] = 1 if vi > hi else (-1 if vi < lo else 0)
            return pd.Series(sig, index=grp.index)

        result[col] = result.groupby("code", group_keys=False).apply(_breakout)
        result = result.sort_values(["date", "code"])
        _log(col, result[col])
        return result

    # ── mean_reversion ────────────────────────────────────────────────────────
    if signal_type == "mean_reversion":
        _require(indicator_cols, "mean_reversion")
        c = indicator_cols[0]
        mr_mode = mode if mode in ("long_only", "short_only", "both") else "long_only"
        col = signal_name or f"signal_meanrev_{c}_n{n}_k{str(k).replace('.','')}"
        result = result.sort_values(["code", "date"])

        def _meanrev(grp):
            v = grp[c].values.astype(float)
            nr = len(v)
            sig = np.zeros(nr, dtype=int)
            for i in range(n, nr):
                w = v[i - n: i + 1]
                mu = np.nanmean(w)
                sd = np.nanstd(w, ddof=1)
                vi = v[i]
                if np.isnan(vi) or np.isnan(sd) or sd == 0:
                    continue
                if mr_mode == "long_only":
                    sig[i] = 1 if vi < mu - k * sd else 0
                elif mr_mode == "short_only":
                    sig[i] = 1 if vi > mu + k * sd else 0
                else:
                    sig[i] = 1 if vi < mu - k * sd else (-1 if vi > mu + k * sd else 0)
            return pd.Series(sig, index=grp.index)

        result[col] = result.groupby("code", group_keys=False).apply(_meanrev)
        result = result.sort_values(["date", "code"])
        _log(col, result[col])
        return result

    # ── regime (price vs MA) ──────────────────────────────────────────────────
    if signal_type == "regime":
        col = signal_name or f"signal_regime_ma{n}"
        result = result.sort_values(["code", "date"])

        def _regime(grp):
            px = grp[close_col].values.astype(float)
            ma = grp[ma_col].values.astype(float) if ma_col else _rolling_ma(px, n)
            cond = np.where(np.isnan(ma), False, px > ma)
            return pd.Series(cond.astype(int), index=grp.index)

        result[col] = result.groupby("code", group_keys=False).apply(_regime)
        result = result.sort_values(["date", "code"])
        _log(col, result[col])
        return result

    # ── score (cross-sectional composite) ─────────────────────────────────────
    if signal_type == "score":
        _require(indicator_cols, "score")
        k_cols = len(indicator_cols)
        w = np.array(weights if weights else [1 / k_cols] * k_cols, dtype=float)
        w /= w.sum()
        col = signal_name or f"signal_score_top{top_n}"
        score_vals = np.full(len(result), np.nan)
        sig_vals = np.zeros(len(result), dtype=int)

        for dt, idx in result.groupby("date").groups.items():
            idx = list(idx)
            mat = np.column_stack([result.loc[idx, c].values.astype(float)
                                   for c in indicator_cols])
            if normalize:
                for j in range(k_cols):
                    col_j = mat[:, j]
                    mu = np.nanmean(col_j)
                    sd = np.nanstd(col_j, ddof=1)
                    mat[:, j] = (col_j - mu) / sd if sd > 0 else col_j - mu
            scores = mat @ w
            score_vals[idx] = scores
            valid = np.where(~np.isnan(scores))[0]
            if len(valid) == 0:
                continue
            ranked = valid[np.argsort(-scores[valid])]
            selected = ranked[:min(top_n, len(ranked))]
            sig_vals[[idx[i] for i in selected]] = 1

        result[score_col] = score_vals
        result[col] = sig_vals
        _log(col, result[col])
        return result

    # ── td_setup ──────────────────────────────────────────────────────────────
    if signal_type == "td_setup":
        td_mode = mode if mode in ("bull", "bear", "both") else "bull"
        col = signal_name or f"signal_td_setup_{td_mode}"
        result = result.sort_values(["code", "date"])

        def _td(grp):
            bull = grp[setup_bull_col].values
            bear = grp[setup_bear_col].values
            nr = len(bull)
            sig = np.zeros(nr, dtype=int)
            for i in range(nr):
                bear_done = (not pd.isna(bear[i])) and bear[i] == 9
                bull_done = (not pd.isna(bull[i])) and bull[i] == 9
                if td_mode == "bull":
                    sig[i] = 1 if bear_done else 0
                elif td_mode == "bear":
                    sig[i] = 1 if bull_done else 0
                else:
                    sig[i] = 1 if bear_done else (-1 if bull_done else 0)
            if td_window > 0:
                extended = sig.copy()
                for i in range(nr):
                    if sig[i] != 0:
                        end = min(nr, i + td_window)
                        extended[i:end] = sig[i]
                sig = extended
            return pd.Series(sig, index=grp.index)

        result[col] = result.groupby("code", group_keys=False).apply(_td)
        result = result.sort_values(["date", "code"])
        _log(col, result[col])
        return result

    # ── earnings ──────────────────────────────────────────────────────────────
    if signal_type == "earnings":
        if earnings_dates is None:
            raise ValueError("earnings signal requires earnings_dates DataFrame")
        earn_mode = mode if mode in ("post", "pre", "both") else "post"
        col = signal_name or f"signal_earnings_{earn_mode}_pre{pre_window}_post{post_window}"
        result = result.copy()
        result["date"] = pd.to_datetime(result["date"])
        earnings_dates = earnings_dates.copy()
        earnings_dates["earnings_date"] = pd.to_datetime(earnings_dates["earnings_date"])
        sig = np.zeros(len(result), dtype=int)

        for cd in result["code"].unique():
            idx = result.index[result["code"] == cd]
            dates_cd = result.loc[idx, "date"].sort_values().values
            ann = earnings_dates.loc[earnings_dates["code"] == cd, "earnings_date"].values
            if len(ann) == 0:
                continue
            active = np.zeros(len(dates_cd), dtype=bool)
            for a in ann:
                pos_arr = np.where(dates_cd >= a)[0]
                if len(pos_arr) == 0:
                    continue
                pos = pos_arr[0]
                if earn_mode in ("post", "both"):
                    end = min(len(dates_cd), pos + post_window + 1)
                    active[pos:end] = True
                if earn_mode in ("pre", "both") and pre_window > 0:
                    start = max(0, pos - pre_window)
                    active[start:pos] = True
            order = result.loc[idx, "date"].argsort()
            sig[idx[order]] = active.astype(int)

        result[col] = sig
        _log(col, result[col])
        return result

    # ── index_rebalance ───────────────────────────────────────────────────────
    if signal_type == "index_rebalance":
        if rebalance_events is None:
            raise ValueError("index_rebalance requires rebalance_events DataFrame")
        col = signal_name or f"signal_idxreb_pre{pre_window}_post{post_window}"
        result = result.copy()
        result["date"] = pd.to_datetime(result["date"])
        revt = rebalance_events.copy()
        revt["rebalance_date"] = pd.to_datetime(revt["rebalance_date"])
        if direction_filter != "both":
            revt = revt[revt["direction"] == direction_filter]
        sig = np.zeros(len(result), dtype=int)

        for cd in result["code"].unique():
            idx = result.index[result["code"] == cd]
            dates_cd = result.loc[idx, "date"].sort_values().values
            evts = revt.loc[revt["code"] == cd, "rebalance_date"].values
            if len(evts) == 0:
                continue
            active = np.zeros(len(dates_cd), dtype=bool)
            for ev in evts:
                pos_arr = np.where(dates_cd >= ev)[0]
                if len(pos_arr) == 0:
                    continue
                pos = pos_arr[0]
                start = max(0, pos - pre_window)
                end = min(len(dates_cd), pos + post_window + 1)
                active[start:end] = True
            order = result.loc[idx, "date"].argsort()
            sig[idx[order]] = active.astype(int)

        result[col] = sig
        _log(col, result[col])
        return result

    # ── vol_regime ────────────────────────────────────────────────────────────
    if signal_type == "vol_regime":
        vr_mode = mode if mode in ("low", "high") else "low"
        col = signal_name or f"signal_volregime_{vol_method}_{vr_mode}"
        result = result.sort_values(["code", "date"])

        def _volreg(grp):
            px = grp[close_col].values.astype(float)
            nr = len(px)
            ret = np.concatenate([[np.nan], np.diff(np.log(np.where(px <= 0, np.nan, px)))])
            vol = np.full(nr, np.nan)
            for i in range(vol_n, nr):
                w = ret[i - vol_n + 1: i + 1]
                vol[i] = np.nanstd(w, ddof=1) * np.sqrt(252)
            sig = np.zeros(nr, dtype=int)
            for i in range(nr):
                if np.isnan(vol[i]):
                    continue
                if vol_method == "threshold":
                    in_low = vol[i] <= vol_threshold
                else:
                    start_h = max(0, i - hist_n + 1)
                    hv = vol[start_h: i + 1]
                    hv = hv[~np.isnan(hv)]
                    if len(hv) < 5:
                        continue
                    cutoff = np.quantile(hv, pct_threshold)
                    in_low = vol[i] <= cutoff
                sig[i] = int(in_low if vr_mode == "low" else not in_low)
            return pd.Series(sig, index=grp.index)

        result[col] = result.groupby("code", group_keys=False).apply(_volreg)
        result = result.sort_values(["date", "code"])
        _log(col, result[col])
        return result

    # ── macro ─────────────────────────────────────────────────────────────────
    if signal_type == "macro":
        if macro_data is None:
            raise ValueError("macro signal requires macro_data DataFrame")
        col = signal_name or f"signal_macro_{macro_method}"
        result = result.copy()
        result["date"] = pd.to_datetime(result["date"])
        md = macro_data.copy()
        md["date"] = pd.to_datetime(md["date"])
        all_dates = sorted(result["date"].unique())
        macro_vals = np.full(len(all_dates), np.nan)
        for i, dt in enumerate(all_dates):
            past = md.loc[md["date"] <= dt, macro_col].values
            if len(past) > 0:
                macro_vals[i] = past[-1]
        n_dates = len(all_dates)
        date_sig = np.zeros(n_dates, dtype=int)
        ops = {
            ">": np.greater, "<": np.less,
            ">=": np.greater_equal, "<=": np.less_equal,
            "==": np.equal, "!=": np.not_equal,
        }
        op_fn = ops.get(compare_op, np.greater)
        for i in range(n_dates):
            v = macro_vals[i]
            if np.isnan(v):
                continue
            if macro_method == "threshold":
                date_sig[i] = int(op_fn(v, threshold))
            elif macro_method == "trend":
                if i < ma_n:
                    continue
                mv_ma = np.nanmean(macro_vals[max(0, i - ma_n + 1): i + 1])
                date_sig[i] = int(v > mv_ma)
            elif macro_method == "change":
                if i < change_n:
                    continue
                prev = macro_vals[i - change_n]
                if not np.isnan(prev):
                    date_sig[i] = int(op_fn(v - prev, threshold))
        date_map = {dt: date_sig[i] for i, dt in enumerate(all_dates)}
        result[col] = result["date"].map(date_map).fillna(0).astype(int)
        _log(col, result[col])
        return result

    # ── window (extend existing signal) ───────────────────────────────────────
    if signal_type == "window":
        src_col = signal_col or (indicator_cols[0] if indicator_cols else None)
        if src_col is None:
            raise ValueError("window signal requires signal_col or indicator_cols")
        col = signal_name or f"{src_col}_win{extend_window}"
        result = result.sort_values(["code", "date"])

        def _extend(grp):
            src = grp[src_col].fillna(0).values.astype(int)
            nr = len(src)
            out = np.zeros(nr, dtype=int)
            for i in range(nr):
                if src[i] != 0:
                    fill = src[i] if carry_value is None else carry_value
                    end = min(nr, i + extend_window)
                    out[i:end] = fill
            return pd.Series(out, index=grp.index)

        result[col] = result.groupby("code", group_keys=False).apply(_extend)
        result = result.sort_values(["date", "code"])
        _log(col, result[col])
        return result

    raise ValueError(f"Unknown signal_type: {signal_type!r}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _require(cols, stype):
    if not cols:
        raise ValueError(f"{stype} signal requires indicator_cols")


def _signal_cols(signal_cols, indicator_cols, stype):
    cols = signal_cols or indicator_cols
    if not cols or len(cols) < 2:
        raise ValueError(f"{stype} signal requires at least 2 columns in signal_cols/indicator_cols")
    return list(cols)


def _op_slug(op: str) -> str:
    return {">": "gt", "<": "lt", ">=": "gte", "<=": "lte",
            "==": "eq", "!=": "neq"}.get(op, "x")


def _resolve_band(df: pd.DataFrame, band) -> np.ndarray:
    if band is None:
        return np.zeros(len(df))
    if isinstance(band, str) and band in df.columns:
        return df[band].fillna(0).values
    return np.full(len(df), float(band))


def _rolling_ma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.nanmean(x[i - n + 1: i + 1])
    return out


def _log(col: str, s: pd.Series):
    print(f" Generated signal column: {col}, valid signals: {int(s.abs().sum())}")
