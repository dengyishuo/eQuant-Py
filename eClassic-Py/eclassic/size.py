"""Size factor (log market cap) — eClassic add_size equivalent."""

from __future__ import annotations

from typing import Optional

import numpy as np

from eclassic._panel import _resolve_col
from equant.utils.panel import slim_output, validate_panel


def size(
    df,
    cap_col: Optional[str] = None,
    new_col: str = "size",
    append: bool = True,
):
    """Size factor — natural logarithm of market capitalization.

    ``size = log(max(cap, 1e-10))``
    """
    validate_panel(df)
    cap = _resolve_col(df, "cap", cap_col)

    result = df.copy()
    result[new_col] = np.log(np.maximum(result[cap].values, 1e-10))

    return slim_output(result, new_col, append)
