"""Parameter grid scanning — parallel backtest sweep over config parameter combinations.

Workflow:
    1. ``param_grid()``      — build the Cartesian product of candidate values
    2. ``run_param_scan()``  — iterate (optionally in parallel), run one backtest
                               per row, collect metrics
    3. ``rank_param_scan()`` — sort / filter / rank the results table
    4. ``sensitivity_table()`` — one-way sweep: vary one param at a time
"""

from __future__ import annotations

import itertools
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields as dc_fields
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ebacktestcraft.analytics import performance_analysis
from ebacktestcraft.config import BacktestConfig
from ebacktestcraft.engine import run


# ─────────────────────────────────────────────────────────────────────────────
# 1.  param_grid
# ─────────────────────────────────────────────────────────────────────────────

def param_grid(**kwargs: Sequence) -> pd.DataFrame:
    """Build a Cartesian parameter grid.

    Parameters
    ----------
    **kwargs
        Each keyword argument is a parameter name (must be a valid
        ``BacktestConfig`` field) mapped to a list of candidate values.

    Returns
    -------
    pd.DataFrame
        One row per unique parameter combination, one column per parameter.

    Examples
    --------
    >>> grid = param_grid(
    ...     rebalance_cycle=["monthly", "quarterly"],
    ...     fixed_component_sl_ratio=[0.05, 0.10, 0.15],
    ... )
    >>> len(grid)   # 2 * 3 = 6
    6
    """
    if not kwargs:
        raise ValueError("Provide at least one named parameter list.")

    valid_fields = {f.name for f in dc_fields(BacktestConfig)}
    bad = set(kwargs) - valid_fields
    if bad:
        raise ValueError(
            f"Unknown BacktestConfig field(s): {bad}. "
            f"Valid fields: {sorted(valid_fields)}"
        )

    names = list(kwargs.keys())
    values = [list(v) for v in kwargs.values()]
    rows = [dict(zip(names, combo)) for combo in itertools.product(*values)]
    return pd.DataFrame(rows, columns=names)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  run_param_scan
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_METRICS = [
    "total_return_pct",
    "annual_return_pct",
    "annual_volatility_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown_pct",
    "calmar_ratio",
    "win_rate_pct",
]


def run_param_scan(
    df: pd.DataFrame,
    grid: pd.DataFrame,
    base_config: Optional[BacktestConfig] = None,
    weight_col: str = "weight",
    metrics: Optional[List[str]] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    n_jobs: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a backtest for every row of *grid* and collect performance metrics.

    Parameters
    ----------
    df : DataFrame
        Long-format market data panel (same format as ``engine.run()``).
    grid : DataFrame
        Parameter combinations from ``param_grid()`` (or any DataFrame whose
        column names are valid ``BacktestConfig`` fields).
    base_config : BacktestConfig, optional
        Base configuration.  Each grid row's values override the base.
        Defaults to ``BacktestConfig()`` (all defaults).
    weight_col : str
        Column in *df* containing portfolio weights.
    metrics : list of str, optional
        Metric names to extract from ``performance_analysis()``.
        Defaults to ``_DEFAULT_METRICS``.
    risk_free_rate : float
        Annual risk-free rate forwarded to ``performance_analysis()``.
    periods_per_year : int
        Trading periods per year (252 daily, 52 weekly, 12 monthly).
    n_jobs : int
        Number of parallel worker processes.  ``1`` = sequential.
        ``-1`` = all available CPUs.  Uses ``concurrent.futures``.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        Grid columns + ``.scan_id``, one column per metric, ``.error``.
        Rows that failed have ``NaN`` metrics and a non-empty ``.error``.
    """
    if metrics is None:
        metrics = _DEFAULT_METRICS
    if base_config is None:
        base_config = BacktestConfig()
    if not isinstance(grid, pd.DataFrame) or len(grid) == 0:
        raise ValueError("`grid` must be a non-empty DataFrame. Use param_grid().")

    n_combos = len(grid)
    if verbose:
        print(f"Starting parameter scan: {n_combos} combination(s)...")

    def _run_one(i: int, row: dict) -> dict:
        cfg_kwargs = {**base_config.to_dict(), **row}
        try:
            cfg = BacktestConfig(**cfg_kwargs)
            result = run(df.copy(), config=cfg, weight_col=weight_col)
            perf = performance_analysis(
                result.equity_curve,
                transactions=result.transactions,
                init_capital=cfg.init_capital,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
            metric_vals = {m: perf.get(m, np.nan) for m in metrics}
            error = ""
        except Exception:
            metric_vals = {m: np.nan for m in metrics}
            error = traceback.format_exc(limit=3)

        if verbose and error == "":
            sharpe = metric_vals.get("sharpe_ratio", np.nan)
            mdd = metric_vals.get("max_drawdown_pct", np.nan)
            print(
                f"  [{i+1}/{n_combos}] done — "
                f"sharpe={sharpe:.3f}  mdd={mdd:.1f}%"
            )
        elif verbose:
            print(f"  [{i+1}/{n_combos}] ERROR: {error.splitlines()[-1]}")

        return {".scan_id": i, **row, **metric_vals, ".error": error}

    rows_input = [
        (i, {k: v for k, v in zip(grid.columns, grid.iloc[i])})
        for i in range(n_combos)
    ]

    if n_jobs == 1:
        results = [_run_one(i, row) for i, row in rows_input]
    else:
        workers = (
            n_jobs if n_jobs > 0
            else __import__("os").cpu_count() or 1
        )
        results_map: dict = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_one, i, row): i
                for i, row in rows_input
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results_map[idx] = fut.result()
                except Exception as exc:
                    row = rows_input[idx][1]
                    results_map[idx] = {
                        ".scan_id": idx,
                        **row,
                        **{m: np.nan for m in metrics},
                        ".error": str(exc),
                    }
        results = [results_map[i] for i in range(n_combos)]

    out = pd.DataFrame(results)
    # Column order: .scan_id, grid cols, metrics, .error
    metric_cols = [m for m in metrics if m in out.columns]
    col_order = [".scan_id"] + list(grid.columns) + metric_cols + [".error"]
    col_order = [c for c in col_order if c in out.columns]
    out = out[col_order].reset_index(drop=True)

    n_ok = (out[".error"] == "").sum()
    if verbose:
        print(f"Scan complete: {n_ok}/{n_combos} succeeded.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  rank_param_scan
# ─────────────────────────────────────────────────────────────────────────────

def rank_param_scan(
    scan_result: pd.DataFrame,
    by: str = "sharpe_ratio",
    descending: bool = True,
    min_sharpe: Optional[float] = None,
    max_drawdown: Optional[float] = None,
    min_return: Optional[float] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Sort, filter, and rank a parameter scan result table.

    Parameters
    ----------
    scan_result : DataFrame
        Output of ``run_param_scan()``.
    by : str
        Column to rank by.  Default ``"sharpe_ratio"``.
    descending : bool
        ``True`` = higher is better (returns, Sharpe); ``False`` = lower is
        better (drawdown).  Default ``True``.
    min_sharpe : float, optional
        Drop rows with ``sharpe_ratio < min_sharpe``.
    max_drawdown : float, optional
        Drop rows with ``max_drawdown_pct < max_drawdown`` (e.g. ``-20``).
    min_return : float, optional
        Drop rows with ``total_return_pct < min_return``.
    top_n : int, optional
        Return only the top *n* rows after filtering.

    Returns
    -------
    pd.DataFrame
        Filtered, sorted result with ``.rank`` prepended.
    """
    if by not in scan_result.columns:
        raise ValueError(f"Column '{by}' not found in scan_result.")

    out = scan_result[scan_result[".error"] == ""].dropna(subset=[by]).copy()

    if min_sharpe is not None and "sharpe_ratio" in out.columns:
        out = out[out["sharpe_ratio"] >= min_sharpe]
    if max_drawdown is not None and "max_drawdown_pct" in out.columns:
        out = out[out["max_drawdown_pct"] >= max_drawdown]
    if min_return is not None and "total_return_pct" in out.columns:
        out = out[out["total_return_pct"] >= min_return]

    out = out.sort_values(by, ascending=not descending).reset_index(drop=True)

    if top_n is not None:
        out = out.head(top_n)

    out.insert(0, ".rank", range(1, len(out) + 1))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4.  sensitivity_table
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_table(
    df: pd.DataFrame,
    param_ranges: Dict[str, Sequence],
    base_config: Optional[BacktestConfig] = None,
    weight_col: str = "weight",
    metrics: Optional[List[str]] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    verbose: bool = True,
) -> pd.DataFrame:
    """One-way sensitivity analysis: vary one parameter at a time.

    All parameters are held fixed at *base_config* values except the one
    being swept.  Returns a long-format table showing how metrics respond
    to each parameter individually.

    Parameters
    ----------
    df : DataFrame
        Market data panel.
    param_ranges : dict
        ``{param_name: [value1, value2, ...]}`` mapping.
    base_config : BacktestConfig, optional
        Fixed baseline.  Defaults to ``BacktestConfig()``.
    weight_col : str
        Weight column in *df*.
    metrics : list of str, optional
        Metrics to extract.  Defaults to ``_DEFAULT_METRICS``.
    risk_free_rate : float
        Annual risk-free rate.
    periods_per_year : int
        Annualisation factor.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns: ``param_name``, ``param_value``, one column per metric,
        ``.error``.
    """
    if metrics is None:
        metrics = _DEFAULT_METRICS
    if base_config is None:
        base_config = BacktestConfig()

    rows = []
    for param_name, candidates in param_ranges.items():
        if verbose:
            print(f"Sweeping: {param_name} ({len(candidates)} values)")
        for v in candidates:
            cfg_kwargs = {**base_config.to_dict(), param_name: v}
            row: dict = {"param_name": param_name, "param_value": str(v)}
            try:
                cfg = BacktestConfig(**cfg_kwargs)
                result = run(df.copy(), config=cfg, weight_col=weight_col)
                perf = performance_analysis(
                    result.equity_curve,
                    transactions=result.transactions,
                    init_capital=cfg.init_capital,
                    risk_free_rate=risk_free_rate,
                    periods_per_year=periods_per_year,
                )
                for m in metrics:
                    row[m] = perf.get(m, np.nan)
                row[".error"] = ""
            except Exception:
                for m in metrics:
                    row[m] = np.nan
                row[".error"] = traceback.format_exc(limit=3)

            rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Convenience: best_params()
# ─────────────────────────────────────────────────────────────────────────────

def best_params(
    scan_result: pd.DataFrame,
    by: str = "sharpe_ratio",
    descending: bool = True,
) -> Dict[str, Any]:
    """Return the parameter dict for the single best row in *scan_result*.

    Parameters
    ----------
    scan_result : DataFrame
        Output of ``run_param_scan()``.
    by : str
        Metric to optimise.
    descending : bool
        Higher = better when ``True``.

    Returns
    -------
    dict
        Parameter name → value for the best combination.
    """
    ranked = rank_param_scan(scan_result, by=by, descending=descending, top_n=1)
    if len(ranked) == 0:
        raise ValueError("No valid results in scan_result.")

    # Return only the original grid columns (exclude .rank, metrics, .error, .scan_id)
    metric_cols = set(_DEFAULT_METRICS) | {".rank", ".scan_id", ".error"}
    param_cols = [c for c in ranked.columns if c not in metric_cols]
    return ranked[param_cols].iloc[0].to_dict()
