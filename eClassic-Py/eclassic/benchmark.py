"""Benchmark factor — eClassic add_benchmark equivalent."""

from __future__ import annotations

from typing import Optional

import numpy as np

from eclassic._panel import _resolve_col
from equant.utils.panel import slim_output, validate_panel


def benchmark(
    df,
    close_col: Optional[str] = None,
    benchmark_col: Optional[str] = None,
    type: str = "excess",
    new_col: str = "bench",
    append: bool = True,
):
    """Benchmark-relative return.

    Parameters
    ----------
    type : str
        ``"excess"`` = asset_ret - bench_ret, ``"ratio"`` = asset_ret / bench_ret.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    result = df.copy()
    result[new_col] = np.nan

    has_bench = benchmark_col is not None and benchmark_col in df.columns
    if not has_bench:
        # Each asset serves as its own benchmark (flat excess = 0)
        result[new_col] = 0.0
        return slim_output(result, new_col, append)

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        asset = sub[col].values.astype(np.float64)
        bench = sub[benchmark_col].values.astype(np.float64)

        asset_ret = np.full(len(asset), np.nan)
        asset_ret[1:] = (asset[1:] - asset[:-1]) / np.maximum(np.abs(asset[:-1]), 1e-15)

        bench_ret = np.full(len(bench), np.nan)
        bench_ret[1:] = (bench[1:] - bench[:-1]) / np.maximum(np.abs(bench[:-1]), 1e-15)

        if type == "excess":
            result.loc[sub.index, new_col] = asset_ret - bench_ret
        else:
            result.loc[sub.index, new_col] = np.where(
                np.abs(bench_ret) > 1e-10,
                asset_ret / bench_ret,
                np.nan,
            )

    return slim_output(result, new_col, append)
