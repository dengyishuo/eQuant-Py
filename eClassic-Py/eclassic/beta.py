"""Beta factor (rolling market beta) — eClassic add_beta equivalent."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from eclassic._panel import _resolve_col
from equant.utils.panel import slim_output, validate_panel


def beta(
    df,
    close_col: Optional[str] = None,
    benchmark_col: Optional[str] = None,
    n: int = 60,
    new_col: str = "beta",
    append: bool = True,
):
    """Rolling market beta via OLS regression.

    ``beta = Cov(asset_ret, bench_ret) / Var(bench_ret)`` over trailing *n* periods.

    Requires a benchmark return column (e.g., index returns) in the DataFrame.
    If no separate benchmark column, uses the asset's own returns (self-beta = 1).
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    result = df.copy()
    result[new_col] = np.nan

    has_bench = benchmark_col is not None and benchmark_col in df.columns

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        asset = sub[col].values.astype(np.float64)
        rets = np.full(len(asset), np.nan)
        rets[1:] = (asset[1:] - asset[:-1]) / np.abs(asset[:-1] + 1e-15)

        if has_bench:
            bench = sub[benchmark_col].values.astype(np.float64)
            bench_rets = np.full(len(bench), np.nan)
            bench_rets[1:] = (bench[1:] - bench[:-1]) / np.abs(bench[:-1] + 1e-15)
            cov = pd.Series(rets).rolling(n, min_periods=n).cov(pd.Series(bench_rets)).values
            var = pd.Series(bench_rets).rolling(n, min_periods=n).var(ddof=1).values
            result.loc[sub.index, new_col] = cov / (var + 1e-15)
        else:
            result.loc[sub.index, new_col] = 1.0  # self-beta

    return slim_output(result, new_col, append)


def slope(
    df,
    close_col: Optional[str] = None,
    benchmark_col: Optional[str] = None,
    n: int = 60,
    new_col: str = "slope",
    append: bool = True,
):
    """Rolling regression slope (alpha + beta * bench).

    Returns the slope coefficient (beta equivalent) plus intercept.
    Produces ``{new_col}_alpha`` and ``{new_col}_beta``.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    out_cols = [f"{new_col}_alpha", f"{new_col}_beta"]
    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    has_bench = benchmark_col is not None and benchmark_col in df.columns
    if not has_bench:
        raise ValueError("benchmark_col is required for slope computation")

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        asset = sub[col].values.astype(np.float64)
        bench = sub[benchmark_col].values.astype(np.float64)
        rets = np.full(len(asset), np.nan)
        rets[1:] = (asset[1:] - asset[:-1]) / np.abs(asset[:-1] + 1e-15)
        bench_rets = np.full(len(bench), np.nan)
        bench_rets[1:] = (bench[1:] - bench[:-1]) / np.abs(bench[:-1] + 1e-15)

        for i in range(n, len(rets)):
            y = rets[i - n + 1 : i + 1]
            x = bench_rets[i - n + 1 : i + 1]
            valid = ~(np.isnan(y) | np.isnan(x))
            if valid.sum() < 5:
                continue
            # Simple OLS: beta = Cov(x,y) / Var(x), alpha = mean(y) - beta * mean(x)
            xv = x[valid]
            yv = y[valid]
            xm = xv.mean()
            ym = yv.mean()
            sxy = ((xv - xm) * (yv - ym)).sum()
            sxx = ((xv - xm) ** 2).sum()
            if sxx > 1e-15:
                b = sxy / sxx
                a = ym - b * xm
                result.loc[sub.index[i], out_cols[0]] = a
                result.loc[sub.index[i], out_cols[1]] = b

    return slim_output(result, out_cols, append)
