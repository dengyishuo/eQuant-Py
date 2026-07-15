"""Weight schemes — equal_weight / fixed_weight / norm_weight / opt_weight / weight."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from equant.utils.panel import validate_panel


def equal_weight(
    df: pd.DataFrame,
    signal_col: str,
    weight_name: Optional[str] = None,
    zero_na: bool = True,
) -> pd.DataFrame:
    """Assign equal weights (1/n) to stocks with signal = 1 per day.

    Parameters
    ----------
    signal_col : str
        Column where value == 1 marks selected stocks.
    weight_name : str, optional
        Output column name. Auto-generated if None.
    zero_na : bool
        Treat NA / Inf in signal column as 0. Default True.
    """
    validate_panel(df)
    if signal_col not in df.columns:
        raise ValueError(f"Signal column not found: {signal_col}")

    col = weight_name or f"weight_equal_{signal_col}"
    result = df.copy()

    sig = result[signal_col].copy()
    if zero_na:
        sig = sig.fillna(0).replace([np.inf, -np.inf], 0)

    is_sel = (sig == 1)
    n_sel = is_sel.groupby(result["date"]).transform("sum")
    result[col] = np.where((n_sel > 0) & is_sel, 1.0 / n_sel, 0.0)

    _diag_weight(result, col, label="equal")
    return result


def fixed_weight(
    df: pd.DataFrame,
    signal_col: str,
    fixed_weights: Union[dict, pd.DataFrame],
    weight_name: Optional[str] = None,
    normalize_daily: bool = False,
    zero_na: bool = True,
    strict_check: bool = True,
) -> pd.DataFrame:
    """Assign pre-defined fixed weights per stock based on signal.

    Parameters
    ----------
    signal_col : str
        Column where value == 1 marks selected stocks.
    fixed_weights : dict or DataFrame
        Two supported formats:

        * ``dict`` — ``{"AAPL": 0.5, "MSFT": 0.3, ...}``
        * ``DataFrame`` — must have ``code`` and ``weight`` columns.
    weight_name : str, optional
        Output column name. Auto-generated if None.
    normalize_daily : bool
        Re-scale selected weights to sum to 1 each day. Default False.
    zero_na : bool
        Treat NA / Inf in signal column as 0. Default True.
    strict_check : bool
        Raise if selected stock set != fixed_weights stock set on any day.
        Default True.
    """
    validate_panel(df)
    if signal_col not in df.columns:
        raise ValueError(f"Signal column not found: {signal_col}")

    # ── parse fixed_weights ────────────────────────────────────────────────
    if isinstance(fixed_weights, dict):
        w_df = pd.DataFrame(
            {"code": list(fixed_weights.keys()),
             "fixed_weight": list(fixed_weights.values())}
        )
    elif isinstance(fixed_weights, pd.DataFrame):
        if not {"code", "weight"}.issubset(fixed_weights.columns):
            raise ValueError("fixed_weights DataFrame must have 'code' and 'weight' columns")
        w_df = fixed_weights[["code", "weight"]].rename(columns={"weight": "fixed_weight"}).copy()
    else:
        raise TypeError("fixed_weights must be a dict or DataFrame")

    fixed_codes = set(w_df["code"])
    total_fixed = w_df["fixed_weight"].sum()
    if abs(total_fixed - 1) > 1e-6 and not normalize_daily:
        print(f" Warning: fixed weights sum to {total_fixed:.4f}, not 1. "
              "Set normalize_daily=True if daily normalization is needed.")

    col = weight_name or f"weight_fixed_{signal_col}"
    result = df.copy()

    # ── clean signal ──────────────────────────────────────────────────────
    sig = result[signal_col].copy()
    if zero_na:
        sig = sig.fillna(0).replace([np.inf, -np.inf], 0)
    is_sel = (sig == 1)

    # ── strict validation ─────────────────────────────────────────────────
    if strict_check:
        print(" Starting strict validation …")
        for dt, grp in result.groupby("date"):
            sel_codes = set(grp.loc[is_sel[grp.index], "code"])
            if sel_codes != fixed_codes:
                raise ValueError(
                    f"Strict check failed on {dt}: "
                    f"selected={sorted(sel_codes)}, "
                    f"fixed={sorted(fixed_codes)}"
                )
        print(f" Strict validation passed across {result['date'].nunique()} days")

    # ── merge and compute ─────────────────────────────────────────────────
    result = result.merge(w_df, on="code", how="left")
    result["fixed_weight"] = result["fixed_weight"].fillna(0.0)
    base = np.where(is_sel, result["fixed_weight"], 0.0)

    if normalize_daily:
        day_total = pd.Series(base, index=result.index).groupby(result["date"]).transform("sum")
        result[col] = np.where(day_total > 0, base / day_total, 0.0)
    else:
        result[col] = base

    result = result.drop(columns=["fixed_weight"])
    _diag_weight(result, col, label="fixed", normalize_daily=normalize_daily)
    return result


def norm_weight(
    df: pd.DataFrame,
    weight_col: str,
    signal_col: Optional[str] = None,
    norm_method: str = "linear",
    weight_name: Optional[str] = None,
    zero_na: bool = True,
) -> pd.DataFrame:
    """Factor-proportional weights via cross-sectional normalization.

    Parameters
    ----------
    weight_col : str
        Raw factor column used to derive weights.
    signal_col : str, optional
        Only stocks with signal == 1 receive weight; others get 0.
        Filtering happens *before* normalization.
    norm_method : str
        ``"linear"``  — weight = v / sum(v)  (default)
        ``"softmax"`` — weight = exp(v) / sum(exp(v))
    zero_na : bool
        Replace NA / Inf with 0 before normalization. Default True.
    """
    validate_panel(df)
    if weight_col not in df.columns:
        raise ValueError(f"Weight column not found: {weight_col}")
    if norm_method not in ("linear", "softmax"):
        raise ValueError("norm_method must be 'linear' or 'softmax'")

    col = weight_name or f"weight_{weight_col}" + (f"_{signal_col}" if signal_col else "")
    result = df.copy()

    v = result[weight_col].copy().astype(float)
    if zero_na:
        v = v.fillna(0).replace([np.inf, -np.inf], 0)

    # apply signal filter before normalization (same as R)
    if signal_col is not None:
        if signal_col not in result.columns:
            raise ValueError(f"Signal column not found: {signal_col}")
        v = v.where(result[signal_col] == 1, 0.0)

    def _norm(grp_v: pd.Series) -> pd.Series:
        arr = grp_v.values.astype(float)
        if norm_method == "linear":
            total = arr.sum()
            return pd.Series(arr / total if total > 1e-15 else np.zeros_like(arr),
                             index=grp_v.index)
        else:  # softmax
            active = arr != 0
            out = np.zeros_like(arr)
            if active.any():
                e = np.exp(arr[active] - arr[active].max())
                out[active] = e / e.sum()
            return pd.Series(out, index=grp_v.index)

    result[col] = v.groupby(result["date"]).transform(
        lambda g: _norm(g)
    )

    _diag_weight(result, col, label=f"norm ({norm_method})")
    return result


def rank_weight(
    df: pd.DataFrame,
    signal_col: str,
    factor_col: str,
    ascending: bool = False,
    weight_name: Optional[str] = None,
) -> pd.DataFrame:
    """Rank-proportional weights: w_i = (n+1-rank_i) / sum(1..n).

    Higher factor value → higher rank → higher weight when ascending=False.

    Parameters
    ----------
    factor_col : str
        Column used to rank stocks within each day's selected universe.
    ascending : bool
        If True, lower factor value gets higher weight. Default False.
    """
    validate_panel(df)
    for c in [signal_col, factor_col]:
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    col = weight_name or f"weight_rank_{factor_col}_{signal_col}"
    result = df.copy()
    result[col] = 0.0

    for dt, grp in result.groupby("date"):
        sel = grp[grp[signal_col] == 1]
        if len(sel) == 0:
            continue
        n = len(sel)
        ranks = sel[factor_col].rank(ascending=ascending, method="average")
        # invert so rank=1 → highest weight
        inv = (n + 1) - ranks
        w = inv / inv.sum()
        result.loc[w.index, col] = w.values

    _diag_weight(result, col, label="rank")
    return result


def inv_vol_weight(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str,
    window: int = 60,
    annual_factor: int = 252,
    weight_name: Optional[str] = None,
) -> pd.DataFrame:
    """Inverse-volatility weights: w_i = (1/σ_i) / Σ(1/σ_j).

    Lower rolling volatility → higher weight.

    Parameters
    ----------
    return_col : str
        Per-asset return column used to compute rolling volatility.
    window : int
        Rolling lookback for volatility estimation. Default 60.
    annual_factor : int
        Annualisation factor (252 / 52 / 12). Default 252.
    """
    validate_panel(df)
    for c in [signal_col, return_col]:
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    col = weight_name or f"weight_inv_vol_{signal_col}"
    result = df.copy().sort_values(["code", "date"]).reset_index(drop=True)

    # rolling annualised vol per stock (time-series, per code)
    result["_vol"] = (
        result.groupby("code")[return_col]
        .transform(lambda x: x.rolling(window, min_periods=5).std() * np.sqrt(annual_factor))
    )
    result[col] = 0.0
    result = result.sort_values(["date", "code"]).reset_index(drop=True)

    for dt, grp in result.groupby("date"):
        sel = grp[(grp[signal_col] == 1) & grp["_vol"].notna() & (grp["_vol"] > 0)]
        if len(sel) == 0:
            continue
        inv_v = 1.0 / sel["_vol"]
        w = inv_v / inv_v.sum()
        result.loc[w.index, col] = w.values

    result = result.drop(columns=["_vol"])
    _diag_weight(result, col, label="inv_vol")
    return result


def target_vol_weight(
    df: pd.DataFrame,
    weight_col: str,
    return_col: str,
    target_vol: float = 0.10,
    window: int = 60,
    annual_factor: int = 252,
    max_leverage: float = 2.0,
    weight_name: Optional[str] = None,
) -> pd.DataFrame:
    """Scale an existing weight column so portfolio hits a target volatility.

    leverage = target_vol / realized_vol, capped at max_leverage.
    Weights are rescaled each day; sum may be < 1 when vol is high.

    Parameters
    ----------
    weight_col : str
        Existing weight column to rescale.
    return_col : str
        Per-asset return column used to compute realised portfolio vol.
    target_vol : float
        Annualised target volatility (e.g. 0.10 = 10 %). Default 0.10.
    window : int
        Rolling lookback for vol estimation. Default 60.
    max_leverage : float
        Cap on the leverage multiplier. Default 2.0.
    """
    validate_panel(df)
    for c in [weight_col, return_col]:
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    col = weight_name or f"weight_tvol_{weight_col}"
    result = df.copy().sort_values(["date", "code"]).reset_index(drop=True)
    dates = sorted(result["date"].unique())
    result[col] = 0.0

    for i, dt in enumerate(dates):
        day_idx = result.index[result["date"] == dt]
        w_today = result.loc[day_idx, weight_col].values

        if i < window or w_today.sum() < 1e-10:
            result.loc[day_idx, col] = w_today
            continue

        past_dates = dates[i - window: i]
        # build portfolio return series using yesterday's weights
        past = result[result["date"].isin(past_dates)].pivot(
            index="date", columns="code", values=return_col
        ).fillna(0)
        # align weights to past columns
        codes_today = result.loc[day_idx, "code"].values
        w_ser = pd.Series(w_today, index=codes_today)
        w_aligned = w_ser.reindex(past.columns, fill_value=0).values
        port_ret = past.values @ w_aligned
        realized_vol = port_ret.std(ddof=1) * np.sqrt(annual_factor)

        leverage = (target_vol / realized_vol) if realized_vol > 1e-10 else 1.0
        leverage = min(leverage, max_leverage)
        result.loc[day_idx, col] = w_today * leverage

    _diag_weight(result, col, label=f"target_vol({target_vol:.0%})")
    return result


def opt_weight(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str,
    opt_type: str = "min_var",
    window: int = 60,
    alpha: float = 0.05,
    rf: float = 0.0,
    benchmark_col: Optional[str] = None,
    annual_factor: int = 252,
    weight_name: Optional[str] = None,
    fallback: str = "equal",
) -> pd.DataFrame:
    """Optimization-based portfolio weights.

    For each trading date, fits weights over a rolling ``window`` of past
    returns to optimise the chosen objective.

    Parameters
    ----------
    signal_col : str
        Column where value == 1 marks the investable universe each day.
    return_col : str
        Column of per-asset period returns (already computed).
    opt_type : str
        One of ``"min_var"``, ``"min_es"``, ``"min_mdd"``,
        ``"max_calmar"``, ``"max_treynor"``.
    window : int
        Rolling lookback periods used to estimate the return distribution.
    alpha : float
        Tail probability for VaR / ES (default 0.05 → 95 % confidence).
    rf : float
        Annual risk-free rate used in Treynor ratio (default 0).
    benchmark_col : str, optional
        Column of benchmark returns required for ``max_treynor``.
    annual_factor : int
        Periods per year for annualisation (252 daily, 52 weekly, 12 monthly).
    weight_name : str, optional
        Output column name. Auto-generated if None.
    fallback : str
        Weight scheme when optimisation fails or history is too short:
        ``"equal"`` (default) assigns 1/n, ``"zero"`` assigns 0.
    """
    _valid = {"min_var", "min_es", "min_mdd", "max_calmar",
              "max_treynor", "risk_parity", "min_variance", "max_sharpe"}
    if opt_type not in _valid:
        raise ValueError(f"opt_type must be one of {sorted(_valid)}")
    if opt_type == "max_treynor" and benchmark_col is None:
        raise ValueError("max_treynor requires benchmark_col")

    validate_panel(df)
    for c in [signal_col, return_col]:
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")
    if benchmark_col and benchmark_col not in df.columns:
        raise ValueError(f"Benchmark column not found: {benchmark_col}")

    col = weight_name or f"weight_{opt_type}_{signal_col}"
    result = df.copy().sort_values(["date", "code"]).reset_index(drop=True)
    result[col] = 0.0
    dates = sorted(result["date"].unique())

    for i, dt in enumerate(dates):
        day_idx = result.index[result["date"] == dt]
        sel_mask = result.loc[day_idx, signal_col] == 1
        sel_idx = day_idx[sel_mask]
        if len(sel_idx) == 0:
            continue
        codes = result.loc[sel_idx, "code"].values

        # build (window × n_assets) return matrix from past window periods
        past_dates = dates[max(0, i - window): i]
        if len(past_dates) < 5:
            _fallback_fill(result, col, sel_idx, fallback)
            continue

        past = result[result["date"].isin(past_dates) & result["code"].isin(codes)]
        R = (past.pivot(index="date", columns="code", values=return_col)
               .reindex(columns=codes)
               .dropna(axis=0))

        if R.shape[0] < 5 or R.shape[1] < 1:
            _fallback_fill(result, col, sel_idx, fallback)
            continue

        R_arr = R.values.astype(float)
        n = R_arr.shape[1]
        w0 = np.ones(n) / n
        bounds = [(0, 1)] * n
        cons = {"type": "eq", "fun": lambda w: w.sum() - 1}

        bm = None
        if benchmark_col:
            bm_past = result[result["date"].isin(past_dates)].groupby("date")[benchmark_col].first()
            bm = bm_past.reindex(R.index).values.astype(float)

        obj = _make_objective(opt_type, R_arr, alpha, rf, annual_factor, bm)
        try:
            res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"ftol": 1e-9, "maxiter": 500})
            w_opt = res.x if res.success else None
        except Exception:
            w_opt = None

        if w_opt is None or np.any(np.isnan(w_opt)):
            _fallback_fill(result, col, sel_idx, fallback)
        else:
            w_opt = np.clip(w_opt, 0, 1)
            w_opt /= w_opt.sum()
            for j, idx in enumerate(sel_idx):
                code = result.loc[idx, "code"]
                pos = np.where(codes == code)[0]
                result.loc[idx, col] = w_opt[pos[0]] if len(pos) else 0.0

    _diag_weight(result, col, label=opt_type)
    return result


# ── objective factories ────────────────────────────────────────────────────────

def _make_objective(opt_type, R, alpha, rf, annual_factor, bm):
    """Return a scalar minimisation objective f(w)."""
    if opt_type == "min_var":
        cov = np.cov(R.T)
        return lambda w: w @ cov @ w

    if opt_type == "min_es":
        def _es(w):
            pr = R @ w
            cutoff = np.percentile(pr, alpha * 100)
            tail = pr[pr <= cutoff]
            return -tail.mean() if len(tail) else 0.0
        return _es

    if opt_type == "min_mdd":
        def _mdd(w):
            pr = R @ w
            nav = np.cumprod(1 + pr)
            peak = np.maximum.accumulate(nav)
            return ((peak - nav) / np.where(peak == 0, 1, peak)).max()
        return _mdd

    if opt_type == "max_calmar":
        def _neg_calmar(w):
            pr = R @ w
            ann_ret = np.mean(pr) * annual_factor
            nav = np.cumprod(1 + pr)
            peak = np.maximum.accumulate(nav)
            mdd = ((peak - nav) / np.where(peak == 0, 1, peak)).max()
            return -(ann_ret / mdd) if mdd > 1e-10 else 1e10
        return _neg_calmar

    if opt_type == "max_treynor":
        rf_period = rf / annual_factor
        def _neg_treynor(w):
            pr = R @ w
            excess = pr - rf_period
            bm_excess = bm - rf_period
            var_bm = np.var(bm_excess, ddof=1)
            if var_bm < 1e-15:
                return 1e10
            beta = np.cov(excess, bm_excess, ddof=1)[0, 1] / var_bm
            if abs(beta) < 1e-10:
                return 1e10
            return -(np.mean(excess) * annual_factor / beta)
        return _neg_treynor

    if opt_type == "risk_parity":
        cov = np.cov(R.T)
        def _risk_parity(w):
            port_var = w @ cov @ w
            if port_var < 1e-20:
                return 0.0
            rc = w * (cov @ w) / np.sqrt(port_var)   # risk contributions
            target = np.sqrt(port_var) / len(w)
            return np.sum((rc - target) ** 2)
        return _risk_parity

    if opt_type == "min_variance":
        cov = np.cov(R.T)
        return lambda w: w @ cov @ w

    if opt_type == "max_sharpe":
        mu = R.mean(axis=0) * annual_factor
        cov = np.cov(R.T)
        rf_annual = rf
        def _neg_sharpe(w):
            ret = mu @ w - rf_annual
            vol = np.sqrt(w @ cov @ w * annual_factor)
            return -(ret / vol) if vol > 1e-10 else 1e10
        return _neg_sharpe

    raise ValueError(f"Unknown opt_type: {opt_type}")


def _fallback_fill(result, col, sel_idx, fallback):
    n = len(sel_idx)
    result.loc[sel_idx, col] = 1.0 / n if fallback == "equal" and n > 0 else 0.0


def weight(
    df: pd.DataFrame,
    weight_type: str,
    **kwargs,
) -> pd.DataFrame:
    """Dispatcher — single entry point for all weight types.

    Parameters
    ----------
    weight_type : str
        ``"equal"``, ``"fixed"``, or ``"norm"``.
    **kwargs
        Forwarded to the underlying function.

    Examples
    --------
    >>> weight(df, "equal", signal_col="signal_mom_gt_0")
    >>> weight(df, "fixed", signal_col="sig", fixed_weights={"A": 0.6, "B": 0.4})
    >>> weight(df, "norm",  weight_col="mom_20", norm_method="softmax")
    """
    _dispatch = {
        # ── analytical ────────────────────────────────────────────
        "equal":        equal_weight,
        "fixed":        fixed_weight,
        "norm":         norm_weight,
        "rank":         rank_weight,
        "inv_vol":      inv_vol_weight,
        "target_vol":   target_vol_weight,
        # ── optimisation-based ────────────────────────────────────
        "min_var":      opt_weight,
        "min_es":       opt_weight,
        "min_mdd":      opt_weight,
        "max_calmar":   opt_weight,
        "max_treynor":  opt_weight,
        "risk_parity":  opt_weight,
        "min_variance": opt_weight,
        "max_sharpe":   opt_weight,
    }
    if weight_type not in _dispatch:
        raise ValueError(
            f"Unknown weight_type: {weight_type!r}. "
            f"Must be one of: {', '.join(sorted(_dispatch))}"
        )
    _opt_types = {"min_var", "min_es", "min_mdd", "max_calmar",
                  "max_treynor", "risk_parity", "min_variance",
                  "max_sharpe"}
    if weight_type in _opt_types:
        return opt_weight(df, opt_type=weight_type, **kwargs)
    return _dispatch[weight_type](df, **kwargs)


# ── internal diagnostics ──────────────────────────────────────────────────────

def _diag_weight(df: pd.DataFrame, col: str, label: str,
                 normalize_daily: bool = False) -> None:
    daily = df.groupby("date")[col].agg(["sum", lambda x: (x > 0).sum()])
    daily.columns = ["total", "n_sel"]
    total_days = len(daily)
    days_with_sel = (daily["n_sel"] > 0).sum()
    avg_sel = daily["n_sel"].mean()
    valid_sum = (daily["total"] - 1).abs().lt(1e-6).sum()

    print(f" Generated {label} weight column: {col}")
    print(f" Total days: {total_days}, days with selection: {days_with_sel}")
    print(f" Average daily selected stocks: {avg_sel:.2f}")
    if normalize_daily or label in ("equal", "norm (linear)", "norm (softmax)"):
        print(f" Days with weight sum = 1: {valid_sum}/{total_days} "
              f"({100*valid_sum/total_days:.1f}%)")
    else:
        print(f" Fixed weight (unnormalized) avg daily sum: "
              f"{daily['total'].mean():.4f}")
