"""Relative Price Strength — eClassic add_rps equivalent."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from eclassic._panel import _resolve_col
from equant.utils.panel import slim_output, validate_panel


def rps(
    df,
    close_col: Optional[str] = None,
    n: Union[int, Sequence[int]] = (60, 120, 250),
    new_col: str = "rps",
    append: bool = True,
):
    """Relative Price Strength.

    Cross-sectional percentile rank of trailing *n*-period return.

    ``RPS = cs_rank(n-period return)`` in [0, 1].
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
        # First compute per-asset n-period return
        result[cname + "_raw"] = np.nan
        for _code, idx in result.groupby("code", sort=False).groups.items():
            sub = result.loc[idx].sort_values("date")
            vals = sub[col].values.astype(np.float64)
            shifted = np.roll(vals, period)
            shifted[:period] = np.nan
            result.loc[sub.index, cname + "_raw"] = (vals - shifted) / np.maximum(np.abs(shifted), 1e-10)

        # Cross-sectional rank per date
        result[cname] = result.groupby("date")[cname + "_raw"].rank(pct=True)

    # Drop raw intermediate columns
    for period in ns:
        result = result.drop(columns=[f"{new_col}_{period}_raw"])

    return slim_output(result, out_cols, append)
