"""Signal generation — eBacktestCraft add_signal equivalent."""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from equant.utils.panel import validate_panel


def signal(
    df: pd.DataFrame,
    indicator_cols: Optional[Sequence[str]] = None,
    signal_type: str = "threshold",
    threshold: float = 0.0,
    compare_op: str = ">",
    cross_upper: Optional[Union[float, str]] = None,
    cross_lower: Optional[Union[float, str]] = None,
    logic_op: str = "&",
    between_lower: Optional[float] = None,
    between_upper: Optional[float] = None,
    signal_name: Optional[str] = None,
    constant_value: int = 1,
) -> pd.DataFrame:
    """Generate 0/1 trading signals.

    Parameters
    ----------
    df : DataFrame
        Long-format panel.
    indicator_cols : sequence of str
        Indicator column names.
    signal_type : str
        ``"threshold"``, ``"crossover"``, ``"multi_condition"``, ``"constant"``, ``"between"``.
    threshold : float
        Threshold for ``"threshold"`` type.
    compare_op : str
        One of ``">"``, ``"<"``, ``">="``, ``"<="``, ``"=="``, ``"!="``.
    cross_upper, cross_lower : float or str
        Crossover bands.
    logic_op : str
        ``"&"`` (AND) or ``"|"`` (OR) for multi_condition.
    between_lower, between_upper : float
        Range for ``"between"`` type.
    signal_name : str, optional
        Output column name (auto-generated if None).
    constant_value : int
        Value for ``"constant"`` type.
    """
    validate_panel(df)

    if signal_type != "constant":
        if not indicator_cols:
            raise ValueError("Non-constant signal types require indicator_cols")
        missing = set(indicator_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found: {missing}")

    result = df.copy()

    # ── Auto-generate signal name ──────────────────────────────────────────
    if signal_name is None:
        if signal_type == "threshold":
            op_map = {">": "gt", "<": "lt", ">=": "gte", "<=": "lte", "==": "eq", "!=": "neq"}
            signal_name = f"signal_{indicator_cols[0]}_{op_map.get(compare_op, 'x')}_{threshold}"
        elif signal_type == "crossover":
            suffix = "cross_down" if cross_lower is not None else "cross_up"
            signal_name = f"signal_{indicator_cols[0]}_{suffix}"
        elif signal_type == "multi_condition":
            logic = "and" if logic_op == "&" else "or"
            signal_name = f"signal_{'_'.join(indicator_cols)}_{logic}"
        elif signal_type == "constant":
            signal_name = f"signal_constant_{constant_value}"
        elif signal_type == "between":
            signal_name = f"signal_{indicator_cols[0]}_between_{between_lower}_{between_upper}"

    # ── Compute signal ─────────────────────────────────────────────────────
    if signal_type == "constant":
        result[signal_name] = constant_value

    elif signal_type == "threshold":
        ops = {
            ">": lambda a, b: a > b, "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
        }
        op_func = ops[compare_op]
        sig = np.ones(len(result), dtype=bool)
        for col in indicator_cols:
            v = result[col].values.copy()
            v[np.isnan(v) | np.isinf(v)] = np.nan
            cond = op_func(v, threshold)
            cond[np.isnan(cond)] = False
            sig = sig & cond
        result[signal_name] = sig.astype(int)

    elif signal_type == "crossover":
        col = indicator_cols[0]
        v = result[col].fillna(0).values
        upper = _resolve_band(result, cross_upper)
        if cross_lower is not None:
            lower = _resolve_band(result, cross_lower)
            cross = (v < lower) & (np.roll(v, 1) > np.roll(lower, 1))
        else:
            cross = (v > upper) & (np.roll(v, 1) < np.roll(upper, 1))
        cross[0] = False
        result[signal_name] = cross.astype(int)

    elif signal_type == "multi_condition":
        sig = None
        for col in indicator_cols:
            v = result[col].fillna(0).values
            cond = v > 0
            sig = cond if sig is None else (sig & cond if logic_op == "&" else sig | cond)
        result[signal_name] = sig.astype(int) if sig is not None else 0

    elif signal_type == "between":
        col = indicator_cols[0]
        v = result[col].values
        cond = (v >= between_lower) & (v <= between_upper)
        cond = np.where(np.isnan(cond), False, cond)
        result[signal_name] = cond.astype(int)

    n_valid = result[signal_name].sum()
    print(f" Generated signal column: {signal_name}, valid signals: {n_valid}")
    return result


def _resolve_band(df: pd.DataFrame, band: Union[float, str, None]) -> np.ndarray:
    """Resolve a crossover band to a numpy array."""
    if band is None:
        return np.zeros(len(df))
    if isinstance(band, str) and band in df.columns:
        return df[band].fillna(0).values
    return np.full(len(df), float(band))
