"""Momentum factor — eClassic add_mom equivalent."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from eclassic._panel import _resolve_col
from equant.utils.panel import slim_output, sort_panel, validate_panel


def momentum(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: Union[int, Sequence[int]] = (2, 5, 10),
    type: str = "continuous",
    na_pad: bool = True,
    new_col: str = "mom",
    append: bool = True,
) -> pd.DataFrame:
    """Momentum factor(s) — price rate of change over *n* periods.

    Parameters
    ----------
    df : DataFrame
        Long-format panel with ``date``, ``code``, ``name``, and a close price column.
    close_col : str, optional
        Name of the close price column. Defaults to ``"close"``.
    n : int or sequence of int
        Lookback period(s). Multiple values produce multiple columns.
    type : str
        ``"continuous"`` = log return, ``"discrete"`` = simple return.
    na_pad : bool
        If True, pad leading observations with NaN.
    new_col : str
        Prefix for output columns (e.g., ``"mom_5"``).
    append : bool
        If False, return only id columns plus new factor columns.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    ns = [n] if isinstance(n, int) else [int(x) for x in n]
    out_cols = [f"{new_col}_{p}" for p in ns]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for period in ns:
        cname = f"{new_col}_{period}"
        for _code, idx in result.groupby("code", sort=False).groups.items():
            sub = result.loc[idx].sort_values("date")
            vals = sub[col].values.astype(np.float64)
            if type == "continuous":
                # Log return
                result.loc[sub.index, cname] = pd.Series(vals).pct_change(periods=period).values
            else:
                shifted = np.roll(vals, period)
                shifted[:period] = np.nan
                result.loc[sub.index, cname] = (vals - shifted) / np.abs(shifted + 1e-15)
            if not na_pad:
                result.loc[sub.index[:period], cname] = result.loc[sub.index[period], cname]

    return slim_output(result, out_cols, append)
