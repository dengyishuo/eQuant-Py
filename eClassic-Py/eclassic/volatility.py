"""Volatility factor — eClassic add_volatility equivalent."""

from __future__ import annotations

from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd

from eclassic._panel import _resolve_col
from equant.utils.panel import slim_output, validate_panel


def volatility(
    df,
    close_col: Optional[str] = None,
    n: Union[int, Sequence[int]] = 20,
    type: str = "sd",
    trading_days: int = 252,
    new_col: str = "vol",
    append: bool = True,
):
    """Rolling volatility factor.

    Parameters
    ----------
    n : int or sequence
        Lookback period(s).
    type : str
        ``"sd"`` = standard deviation, ``"var"`` = variance.
    trading_days : int
        Annualisation factor. Set to 1 for raw daily vol.
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
            rets = np.full(len(vals), np.nan)
            rets[1:] = vals[1:] / np.maximum(np.abs(vals[:-1]), 1e-15) - 1.0

            rolled = pd.Series(rets).rolling(window=period, min_periods=period)
            if type == "sd":
                result.loc[sub.index, cname] = rolled.std(ddof=1).values * np.sqrt(trading_days)
            else:
                result.loc[sub.index, cname] = rolled.var(ddof=1).values * trading_days

    return slim_output(result, out_cols, append)
