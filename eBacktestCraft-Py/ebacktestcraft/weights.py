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
        ``"max_calmar"`` (alias ``"max_kama"``),
        ``"max_treynor"`` (alias ``"max_terino"``).
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
    _valid = {"min_var", "min_es", "min_mdd", "max_calmar", "max_kama",
              "max_treynor", "max_terino"}
    if opt_type not in _valid:
        raise ValueError(f"opt_type must be one of {sorted(_valid)}")
    # normalise aliases
    if opt_type == "max_kama":
        opt_type = "max_calmar"
    if opt_type == "max_terino":
        opt_type = "max_treynor"
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
        "equal":       equal_weight,
        "fixed":       fixed_weight,
        "norm":        norm_weight,
        "min_var":     opt_weight,
        "min_es":      opt_weight,
        "min_mdd":     opt_weight,
        "max_calmar":  opt_weight,
        "max_kama":    opt_weight,
        "max_treynor": opt_weight,
        "max_terino":  opt_weight,
    }
    if weight_type not in _dispatch:
        raise ValueError(
            f"Unknown weight_type: {weight_type!r}. "
            f"Must be one of: {', '.join(_dispatch)}"
        )
    # opt_weight types need opt_type forwarded
    _opt_types = {"min_var", "min_es", "min_mdd",
                  "max_calmar", "max_kama", "max_treynor", "max_terino"}
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
