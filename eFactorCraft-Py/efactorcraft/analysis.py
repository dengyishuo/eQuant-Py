"""Factor evaluation — eFactorCraft ic_analysis / ir_analysis / quantile_analysis."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from equant.utils.panel import slim_output, validate_panel


def add_next_return(
    df: pd.DataFrame,
    close_col: str = "close",
    periods: Union[int, Sequence[int]] = (1, 5, 20),
    new_col: str = "forward",
    append: bool = True,
) -> pd.DataFrame:
    """Add forward returns for IC/quantile analysis.

    Parameters
    ----------
    periods : int or sequence
        Forward horizon(s). Multiple values produce multiple columns.
    """
    validate_panel(df)
    ps = [periods] if isinstance(periods, int) else list(periods)
    out_cols = [f"{new_col}_{p}" for p in ps]
    result = df.copy()

    for p in ps:
        cname = f"{new_col}_{p}"
        result[cname] = np.nan
        for _code, idx in result.groupby("code", sort=False).groups.items():
            sub = result.loc[idx].sort_values("date")
            vals = sub[close_col].values.astype(np.float64)
            shifted = np.roll(vals, -p)
            shifted[-p:] = np.nan
            result.loc[sub.index, cname] = (shifted - vals) / np.maximum(np.abs(vals), 1e-15)

    return slim_output(result, out_cols, append)


def ic_analysis(
    df: pd.DataFrame,
    factor_cols: Union[str, Sequence[str]],
    forward_col: Optional[str] = None,
    forward_cols: Optional[Sequence[str]] = None,
    method: str = "pearson",
) -> dict:
    """Information Coefficient analysis.

    Computes the cross-sectional correlation between each factor and
    each forward return, per date.

    Returns a dict keyed by factor name, each value a DataFrame with
    dates as index and forward periods as columns.
    """
    validate_panel(df)
    factors = [factor_cols] if isinstance(factor_cols, str) else list(factor_cols)

    # Determine forward return columns
    if forward_cols is not None:
        fwd_cols = list(forward_cols)
    elif forward_col is not None:
        fwd_cols = [forward_col]
    else:
        fwd_cols = [c for c in df.columns if c.startswith("forward_")]
    if not fwd_cols:
        raise ValueError("No forward return columns found. Run add_next_return() first.")

    results = {}

    for factor in factors:
        ic_table = {}
        for fwd in fwd_cols:
            def _ic(group):
                return group[factor].corr(group[fwd], method=method)

            ic_series = df.groupby("date").apply(_ic, include_groups=False)
            period_label = fwd.replace("forward_", "fwd")
            ic_table[period_label] = ic_series

        results[factor] = pd.DataFrame(ic_table)

    return results


def ir_analysis(ic_results: dict) -> pd.DataFrame:
    """Information Ratio from IC results.

    ``IR = mean(IC) / std(IC)`` per factor per forward horizon.
    """
    rows = []
    for factor, ic_df in ic_results.items():
        row = {"factor": factor}
        for col in ic_df.columns:
            ic = ic_df[col].dropna()
            if len(ic) < 2:
                row[f"IR_{col}"] = np.nan
            else:
                row[f"IR_{col}"] = ic.mean() / ic.std(ddof=1)
        rows.append(row)

    return pd.DataFrame(rows).set_index("factor")


def quantile_analysis(
    df: pd.DataFrame,
    factor_col: str,
    forward_col: str = "forward_1",
    n_groups: int = 10,
    by: str = "date",
) -> pd.DataFrame:
    """Quantile (decile) analysis — mean forward return per group.

    Assigns assets to *n_groups* bins per date, then aggregates
    mean forward return per group.
    """
    validate_panel(df)
    result = df.copy()

    # Assign quantile groups
    def _qcut(g):
        try:
            return pd.qcut(g, n_groups, labels=False, duplicates="drop") + 1
        except ValueError:
            return pd.Series(np.nan, index=g.index)

    result["_qgroup"] = result.groupby(by)[factor_col].transform(_qcut)

    # Mean forward return per group per date
    qa = result.groupby(["date", "_qgroup"])[forward_col].mean().reset_index()
    qa = qa.rename(columns={"_qgroup": "group", forward_col: "mean_return"})

    # Aggregate across time: mean per group
    summary = qa.groupby("group")["mean_return"].agg(["mean", "std", "count"]).reset_index()

    # Long-short spread
    if 1 in summary["group"].values and n_groups in summary["group"].values:
        top = summary.loc[summary["group"] == n_groups, "mean"].values[0]
        bottom = summary.loc[summary["group"] == 1, "mean"].values[0]
        spread = top - bottom
    else:
        spread = np.nan

    return {"by_date": qa, "summary": summary, "long_short_spread": spread}
