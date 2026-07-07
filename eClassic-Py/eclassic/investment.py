"""Investment factor (asset growth) — eClassic add_investment equivalent."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from eclassic._panel import _resolve_col
from equant.utils.panel import slim_output, validate_panel


def investment(
    df,
    assets_col: Optional[str] = None,
    n: int = 252,
    new_col: str = "investment",
    append: bool = True,
):
    """Investment factor — year-over-year total asset growth.

    ``investment = (assets[t] - assets[t-n]) / |assets[t-n]|``

    Parameters
    ----------
    n : int
        Lag in trading days. Default 252 ≈ 1 year.
    """
    validate_panel(df)
    col = _resolve_col(df, "assets", assets_col)

    result = df.copy()
    result[new_col] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        vals = sub[col].values.astype(np.float64)
        shifted = np.roll(vals, n)
        shifted[:n] = np.nan
        result.loc[sub.index, new_col] = (vals - shifted) / np.maximum(np.abs(shifted), 1e-10)

    return slim_output(result, new_col, append)
