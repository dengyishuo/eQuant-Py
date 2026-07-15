"""Enhanced signal types — smarter entry/exit logic.

Each function takes a long-format panel and appends a 0/1 signal column.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from equant.utils.panel import slim_output, validate_panel


# ══════════════════════════════════════════════════════════════════════════════
# quantile_signal — buy the top fraction
# ══════════════════════════════════════════════════════════════════════════════


def quantile_signal(
    df: pd.DataFrame,
    indicator_col: str,
    top_frac: float = 0.2,
    higher_is_better: bool = True,
    signal_name: Optional[str] = None,
    long_only: bool = True,
    append: bool = True,
) -> pd.DataFrame:
    """Buy assets in the top (or bottom) quantile of an indicator.

    Parameters
    ----------
    indicator_col : str
        Factor/indicator column to rank.
    top_frac : float
        Fraction of assets to select (0.1 = top 10%, 0.5 = top half).
    higher_is_better : bool
        If True, select assets with the highest indicator values.
        If False, select the lowest.
    long_only : bool
        If False, also short the bottom top_frac (returns -1, 0, +1).
    """
    validate_panel(df, extra_cols=[indicator_col])
    result = df.copy()

    if signal_name is None:
        signal_name = f"signal_q_{indicator_col}_top{int(top_frac*100)}"

    signal_vals = np.zeros(len(result), dtype=int)

    for date, idx in result.groupby("date", sort=False).groups.items():
        vals = result.loc[idx, indicator_col].values.astype(np.float64)
        valid_mask = ~np.isnan(vals)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) < 3:
            continue

        n_select = max(1, int(len(valid_idx) * top_frac))
        if higher_is_better:
            top_n = np.argpartition(-vals[valid_mask], n_select - 1)[:n_select]
        else:
            top_n = np.argpartition(vals[valid_mask], n_select - 1)[:n_select]

        selected_positions = valid_idx[top_n]
        abs_idx = idx[selected_positions]
        signal_vals[abs_idx] = 1

        if not long_only:
            # Also short bottom fraction
            if higher_is_better:
                bottom_n = np.argpartition(vals[valid_mask], n_select - 1)[:n_select]
            else:
                bottom_n = np.argpartition(-vals[valid_mask], n_select - 1)[:n_select]
            short_positions = valid_idx[bottom_n]
            abs_idx_short = idx[short_positions]
            signal_vals[abs_idx_short] = -1

    result[signal_name] = signal_vals
    return slim_output(result, signal_name, append)


# ══════════════════════════════════════════════════════════════════════════════
# persistent_signal — require consecutive days
# ══════════════════════════════════════════════════════════════════════════════


def persistent_signal(
    df: pd.DataFrame,
    indicator_col: str,
    threshold: float = 0.0,
    compare_op: str = ">",
    min_days: int = 3,
    signal_name: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Signal only after a condition persists for min_days consecutive days.

    Filters out false breakouts by requiring the threshold condition to
    hold for multiple consecutive days before generating a signal.

    Parameters
    ----------
    min_days : int
        Number of consecutive days the condition must hold.
    """
    validate_panel(df, extra_cols=[indicator_col])
    result = df.copy()

    ops = {">": np.greater, "<": np.less, ">=": np.greater_equal,
           "<=": np.less_equal, "==": np.equal, "!=": np.not_equal}
    op_fn = ops.get(compare_op, np.greater)

    if signal_name is None:
        op_map = {">": "gt", "<": "lt", ">=": "gte", "<=": "lte", "==": "eq", "!=": "neq"}
        signal_name = f"signal_p_{indicator_col}_{op_map.get(compare_op,'gt')}_{threshold}_d{min_days}"

    signal_vals = np.zeros(len(result), dtype=int)

    for code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, indicator_col].values.astype(np.float64)
        # Raw condition: True/False each day
        raw = op_fn(vals, threshold).astype(int)
        # Count consecutive True days
        streak = np.zeros(len(vals), dtype=int)
        count = 0
        for i in range(len(vals)):
            if raw[i] == 1 and not np.isnan(vals[i]):
                count += 1
            else:
                count = 0
            streak[i] = count
        # Signal = 1 when streak >= min_days
        sig = (streak >= min_days).astype(int)
        # But turn off signal if raw condition breaks
        sig = sig * raw
        signal_vals[idx] = sig

    result[signal_name] = signal_vals
    return slim_output(result, signal_name, append)


# ══════════════════════════════════════════════════════════════════════════════
# smoothed_signal — use moving average of indicator to reduce noise
# ══════════════════════════════════════════════════════════════════════════════


def smoothed_signal(
    df: pd.DataFrame,
    indicator_col: str,
    smooth_period: int = 5,
    threshold: float = 0.0,
    compare_op: str = ">",
    signal_name: Optional[str] = None,
    append: bool = True,
) -> pd.DataFrame:
    """Signal based on a smoothed (rolling-mean) version of the indicator.

    Reduces noise-trading by requiring the moving average of the indicator
    to cross the threshold, rather than the raw daily value.

    Parameters
    ----------
    smooth_period : int
        Rolling window for smoothing.
    """
    validate_panel(df, extra_cols=[indicator_col])
    result = df.copy()

    ops = {">": np.greater, "<": np.less, ">=": np.greater_equal,
           "<=": np.less_equal, "==": np.equal, "!=": np.not_equal}
    op_fn = ops.get(compare_op, np.greater)

    if signal_name is None:
        op_map = {">": "gt", "<": "lt", ">=": "gte", "<=": "lte", "==": "eq", "!=": "neq"}
        signal_name = f"signal_sm{smooth_period}_{indicator_col}_{op_map.get(compare_op,'gt')}_{threshold}"

    signal_vals = np.zeros(len(result), dtype=int)
    smooth_col = f"_smoothed_{indicator_col}"
    result[smooth_col] = np.nan

    for code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, indicator_col].values.astype(np.float64)
        smoothed = pd.Series(vals).rolling(smooth_period, min_periods=1).mean().values
        result.loc[idx, smooth_col] = smoothed
        sig = op_fn(smoothed, threshold).astype(int)
        signal_vals[idx] = sig

    result[signal_name] = signal_vals
    result.drop(columns=[smooth_col], inplace=True)
    return slim_output(result, signal_name, append)
