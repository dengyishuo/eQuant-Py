"""Profitability factor (ROE-style) — eClassic add_profitability equivalent."""

from __future__ import annotations

from typing import Optional

import numpy as np

from eclassic._panel import _resolve_col
from equant.utils.panel import slim_output, validate_panel


def profitability(
    df,
    op_col: Optional[str] = None,
    bv_col: Optional[str] = None,
    new_col: str = "profitability",
    append: bool = True,
):
    """Profitability factor = operating profit / book value.

    ``profitability = op / max(|bv|, 1e-10)``
    """
    validate_panel(df)
    op = _resolve_col(df, "op", op_col)
    bv = _resolve_col(df, "bv", bv_col)

    result = df.copy()
    result[new_col] = np.where(
        np.abs(result[bv].values) < 1e-10,
        np.nan,
        result[op].values / np.maximum(np.abs(result[bv].values), 1e-10),
    )

    return slim_output(result, new_col, append)
