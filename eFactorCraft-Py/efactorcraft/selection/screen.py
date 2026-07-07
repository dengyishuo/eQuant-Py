"""Factor selection — screen and rank factors by effectiveness and stability.

These are analysis functions that return structured data (dicts or DataFrames),
following the pattern established by ``eng.ic_analysis`` and ``eng.ir_analysis``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from equant.utils.panel import validate_panel


def _compute_factor_ics(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    forward_col: str,
) -> dict[str, pd.Series]:
    """Compute IC per date for each factor. Returns {factor_name: Series(index=date)}."""
    dates = sorted(df["date"].unique())
    ic_data: dict[str, list[float]] = {fc: [] for fc in factor_cols}
    date_labels = []
    for d in dates:
        sub = df[df["date"] == d]
        date_labels.append(d)
        for fc in factor_cols:
            valid = sub[[fc, forward_col]].dropna()
            if len(valid) < 5:
                ic_data[fc].append(np.nan)
            else:
                ic_data[fc].append(valid[fc].corr(valid[forward_col]))
    return {fc: pd.Series(ic_data[fc], index=pd.DatetimeIndex(date_labels)) for fc in factor_cols}


def _compute_factor_ir(ic_series: pd.Series) -> float:
    """Information Ratio = mean(IC) / std(IC)."""
    vals = ic_series.dropna()
    if len(vals) < 5:
        return 0.0
    return float(vals.mean() / max(vals.std(), 1e-10))


# ══════════════════════════════════════════════════════════════════════════════
# ic_screen
# ══════════════════════════════════════════════════════════════════════════════


def ic_screen(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    forward_col: str = "forward_1",
    min_abs_ic: float = 0.02,
    min_ir: float = 0.3,
) -> dict:
    """Filter factors by minimum absolute IC and IR thresholds.

    Returns
    -------
    dict
        ``{"passed": list[str], "failed": list[str], "report": DataFrame}``
    """
    validate_panel(df, extra_cols=list(factor_cols))
    ic_data = _compute_factor_ics(df, factor_cols, forward_col)

    rows = []
    passed, failed = [], []
    for fc in factor_cols:
        ic_mean = ic_data[fc].mean()
        ic_std = ic_data[fc].std()
        ir = _compute_factor_ir(ic_data[fc])
        ok = abs(ic_mean) >= min_abs_ic and abs(ir) >= min_ir
        (passed if ok else failed).append(fc)
        rows.append({
            "factor": fc, "ic_mean": round(ic_mean, 6), "ic_std": round(ic_std, 6),
            "ir": round(ir, 4), "passed": ok,
        })

    report = pd.DataFrame(rows).sort_values("ir", key=abs, ascending=False).reset_index(drop=True)
    return {"passed": passed, "failed": failed, "report": report}


# ══════════════════════════════════════════════════════════════════════════════
# correlation_screen
# ══════════════════════════════════════════════════════════════════════════════


def correlation_screen(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    max_corr: float = 0.7,
    forward_col: str = "forward_1",
    method: str = "pearson",
) -> dict:
    """Remove redundant highly-correlated factors.

    For each pair with |avg_correlation| > max_corr, the factor with the
    lower absolute IR is dropped.

    Returns
    -------
    dict
        ``{"kept": list[str], "removed": list[str], "corr_matrix": DataFrame}``
    """
    validate_panel(df, extra_cols=list(factor_cols))
    dates = sorted(df["date"].unique())

    # Average pairwise correlation over time
    all_corrs = []
    for d in dates:
        sub = df[df["date"] == d][list(factor_cols)].dropna()
        if len(sub) < 5:
            continue
        all_corrs.append(sub.corr(method=method).values)
    if not all_corrs:
        return {"kept": list(factor_cols), "removed": [], "corr_matrix": pd.DataFrame()}

    avg_corr = np.mean(all_corrs, axis=0)
    corr_df = pd.DataFrame(avg_corr, index=list(factor_cols), columns=list(factor_cols))

    # Compute IR for tie-breaking
    ic_data = _compute_factor_ics(df, factor_cols, forward_col)
    irs = {fc: abs(_compute_factor_ir(ic_data[fc])) for fc in factor_cols}

    # Greedy removal
    removed: set[str] = set()
    fc_list = list(factor_cols)
    for i in range(len(fc_list)):
        for j in range(i + 1, len(fc_list)):
            fi, fj = fc_list[i], fc_list[j]
            if fi in removed or fj in removed:
                continue
            if abs(avg_corr[i, j]) > max_corr:
                # Drop the factor with lower absolute IR
                if irs.get(fi, 0) < irs.get(fj, 0):
                    removed.add(fi)
                else:
                    removed.add(fj)

    kept = [fc for fc in factor_cols if fc not in removed]
    return {"kept": kept, "removed": list(removed), "corr_matrix": corr_df}


# ══════════════════════════════════════════════════════════════════════════════
# stability_screen
# ══════════════════════════════════════════════════════════════════════════════


def stability_screen(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    forward_col: str = "forward_1",
    window: int = 60,
) -> pd.DataFrame:
    """Rank factors by IC stability (lower IC turnover = more stable).

    IC turnover = std(IC_diff) over the trailing window, where
    IC_diff(t) = IC(t) - IC(t-1).

    Returns
    -------
    DataFrame
        Columns: ``factor``, ``ic_mean``, ``ic_std``, ``ir``, ``ic_turnover``,
        ``stability_rank``.
    """
    validate_panel(df, extra_cols=list(factor_cols))
    ic_data = _compute_factor_ics(df, factor_cols, forward_col)

    rows = []
    for fc in factor_cols:
        ic = ic_data[fc].dropna()
        if len(ic) < 10:
            rows.append({
                "factor": fc, "ic_mean": np.nan, "ic_std": np.nan,
                "ir": np.nan, "ic_turnover": np.nan, "stability_rank": 999,
            })
            continue
        ic_diff = ic.diff().dropna()
        if len(ic_diff) < min(window, len(ic_diff)):
            turnover = ic_diff.std()
        else:
            turnover = ic_diff.iloc[-window:].std()
        rows.append({
            "factor": fc,
            "ic_mean": round(float(ic.mean()), 6),
            "ic_std": round(float(ic.std()), 6),
            "ir": round(_compute_factor_ir(ic), 4),
            "ic_turnover": round(float(turnover), 6),
            "stability_rank": 0,  # filled below
        })

    report = pd.DataFrame(rows)
    report["stability_rank"] = report["ic_turnover"].rank(ascending=True)
    return report.sort_values("stability_rank").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# select_top
# ══════════════════════════════════════════════════════════════════════════════


def select_top(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    forward_col: str = "forward_1",
    top_n: int = 10,
    criterion: str = "ir",
) -> list[str]:
    """Select top-N factors by a criterion.

    Parameters
    ----------
    criterion : str
        ``"ir"`` (absolute IR), ``"ic"`` (absolute mean IC), or
        ``"stability"`` (low IC turnover).
    """
    if top_n > len(factor_cols):
        raise ValueError(f"top_n ({top_n}) > number of factor_cols ({len(factor_cols)})")

    validate_panel(df, extra_cols=list(factor_cols))
    st = stability_screen(df, factor_cols, forward_col)

    if criterion == "ir":
        st = st.sort_values("ir", key=abs, ascending=False)
    elif criterion == "ic":
        st = st.sort_values("ic_mean", key=abs, ascending=False)
    elif criterion == "stability":
        st = st.sort_values("ic_turnover", ascending=True)
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Use 'ir', 'ic', or 'stability'.")

    return st["factor"].iloc[:top_n].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# factor_report
# ══════════════════════════════════════════════════════════════════════════════


def factor_report(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    forward_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Comprehensive factor evaluation report.

    Returns a DataFrame indexed by factor name with IC, IR, and stability
    metrics for each forward-return horizon.

    Parameters
    ----------
    forward_cols : sequence of str, optional
        Forward return columns. Auto-detects ``forward_*`` columns if None.
    """
    validate_panel(df, extra_cols=list(factor_cols))
    if forward_cols is None:
        forward_cols = [c for c in df.columns if c.startswith("forward_")]
        if not forward_cols:
            raise ValueError("No forward_* columns found. Run eng.add_next_return() first.")

    all_rows = []
    for fc in factor_cols:
        row: dict = {"factor": fc}
        for fwd in forward_cols:
            ic_vals = _compute_factor_ics(df, [fc], fwd)[fc].dropna()
            if len(ic_vals) < 5:
                row[f"{fwd}_ic_mean"] = np.nan
                row[f"{fwd}_ic_std"] = np.nan
                row[f"{fwd}_ir"] = np.nan
            else:
                row[f"{fwd}_ic_mean"] = round(float(ic_vals.mean()), 6)
                row[f"{fwd}_ic_std"] = round(float(ic_vals.std()), 6)
                row[f"{fwd}_ir"] = round(_compute_factor_ir(ic_vals), 4)

        # Average correlation against other factors
        other = [f for f in factor_cols if f != fc]
        if other:
            cors = []
            for d in sorted(df["date"].unique()):
                sub = df[df["date"] == d][[fc] + other].dropna()
                if len(sub) < 5:
                    continue
                avg_c = sub[fc].corr(sub[other].mean(axis=1)) if len(other) > 0 else 1.0
                cors.append(avg_c)
            row["avg_pairwise_corr"] = round(float(np.mean(cors)) if cors else np.nan, 4)
        else:
            row["avg_pairwise_corr"] = np.nan

        all_rows.append(row)

    report = pd.DataFrame(all_rows)
    if forward_cols:
        first_fwd = forward_cols[0]
        ir_col = f"{first_fwd}_ir"
        if ir_col in report.columns:
            report = report.sort_values(ir_col, key=abs, ascending=False)
    return report.reset_index(drop=True)
