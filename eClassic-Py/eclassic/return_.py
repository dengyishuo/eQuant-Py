"""Return factor — eClassic add_return equivalent."""

from __future__ import annotations

from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd

from eclassic._panel import _resolve_col
from equant.utils.panel import slim_output, validate_panel


def return_(
    df,
    close_col: Optional[str] = None,
    n: Union[int, Sequence[int]] = 1,
    type: str = "continuous",
    na_pad: bool = True,
    new_col: str = "ret",
    append: bool = True,
):
    """Forward/backward return over *n* periods.

    Note: named ``return_`` to avoid shadowing the Python keyword.
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
                # Log-forward return
                shifted = np.roll(vals, -period)
                shifted[-period:] = np.nan
                result.loc[sub.index, cname] = (shifted - vals) / np.maximum(np.abs(vals), 1e-15)
            else:
                result.loc[sub.index, cname] = pd.Series(vals).pct_change(periods=-period).values

    return slim_output(result, out_cols, append)
