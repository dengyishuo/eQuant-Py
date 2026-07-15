"""SMA factor — eClassic add_sma equivalent."""

from __future__ import annotations

from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd

from eclassic._panel import _resolve_col
from equant.utils.panel import slim_output, validate_panel


def sma(
    df,
    close_col: Optional[str] = None,
    n: Union[int, Sequence[int]] = 20,
    new_col: str = "SMA",
    append: bool = True,
):
    """Simple Moving Average factor.

    ``SMA_n = SMA(close, n) / close`` — deviation of price from its moving average.
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
            sma_vals = pd.Series(vals).rolling(window=period, min_periods=period).mean().values
            result.loc[sub.index, cname] = sma_vals / np.maximum(np.abs(vals), 1e-10) - 1.0

    return slim_output(result, out_cols, append)
