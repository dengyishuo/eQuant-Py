"""Value factor (B/M ratio) — eClassic add_value equivalent."""

from __future__ import annotations

from typing import Optional

import numpy as np

from eclassic._panel import _resolve_col
from equant.utils.panel import slim_output, validate_panel


def value(
    df,
    bv_col: Optional[str] = None,
    cap_col: Optional[str] = None,
    new_col: str = "value",
    append: bool = True,
):
    """Book-to-Market value factor.

    ``value = book_value / market_cap``
    Returns NaN when market cap is negligibly small.
    """
    validate_panel(df)
    bv = _resolve_col(df, "bv", bv_col)
    cap = _resolve_col(df, "cap", cap_col)

    result = df.copy()
    result[new_col] = np.where(
        np.abs(result[cap].values) < 1e-10,
        np.nan,
        result[bv].values / result[cap].values,
    )

    return slim_output(result, new_col, append)
