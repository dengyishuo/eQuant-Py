"""Benchmark strategies — eBacktestCraft run_benchmark_* equivalent."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ebacktestcraft.config import BacktestConfig
from ebacktestcraft.engine import run
from equant.utils.panel import validate_panel


def equal_weight_benchmark(
    df: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    **kwargs,
):
    """Run a 1/N equal-weight benchmark strategy.

    All assets get equal weight every rebalance period. No signals.
    """
    validate_panel(df)
    result = df.copy()
    result["_bench_signal"] = 1  # Always selected
    result["weight"] = np.nan

    for date in result["date"].unique():
        day_mask = result["date"] == date
        n_assets = day_mask.sum()
        if n_assets > 0:
            result.loc[day_mask, "weight"] = 1.0 / n_assets

    return run(result, config=config, weight_col="weight", **kwargs)


def buy_and_hold_benchmark(
    df: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    **kwargs,
):
    """Run a buy-and-hold benchmark.

    Disables rebalancing so each asset's initial allocation is held
    until the end.
    """
    cfg = config or BacktestConfig()
    cfg.rebalance_mode = "calendar"
    cfg.rebalance_cycle = "99999"  # Effectively never rebalances
    if kwargs:
        cfg.set(**kwargs)

    return equal_weight_benchmark(df, config=cfg)


def index_benchmark(
    df: pd.DataFrame,
    index_code: str,
    config: Optional[BacktestConfig] = None,
    **kwargs,
):
    """Run a benchmark that tracks a single index/ETF.

    Parameters
    ----------
    df : DataFrame
        Must contain the index asset.
    index_code : str
        Code of the index asset.
    """
    validate_panel(df)
    result = df.copy()
    result["_bench_signal"] = (result["code"] == index_code).astype(int)
    result["weight"] = np.where(result["code"] == index_code, 1.0, 0.0)
    return run(result, config=config, weight_col="weight", **kwargs)


def compare_benchmarks(
    results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare multiple benchmark equity curves.

    Parameters
    ----------
    results : dict
        Mapping of strategy name to equity_curve DataFrame.
    """
    comparisons = []
    for name, eq in results.items():
        nav = eq["nav"].values
        total_ret = (nav[-1] / nav[0] - 1) * 100 if len(nav) > 1 else 0.0
        daily_rets = eq["return"].dropna().values if "return" in eq.columns else np.diff(nav) / nav[:-1]
        vol = np.std(daily_rets, ddof=1) * np.sqrt(252) * 100 if len(daily_rets) > 1 else 0.0
        sharpe = np.sqrt(252) * daily_rets.mean() / daily_rets.std() if daily_rets.std() > 0 else 0.0

        cummax = np.maximum.accumulate(nav)
        dd = (nav - cummax) / cummax
        max_dd = dd.min() * 100

        comparisons.append({
            "strategy": name,
            "total_return_pct": total_ret,
            "annual_vol_pct": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd,
        })

    return pd.DataFrame(comparisons)
