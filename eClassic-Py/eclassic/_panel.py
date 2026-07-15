"""Shared panel-data utilities for the factors module."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from equant.utils.panel import slim_output, sort_panel, validate_panel


def _resolve_col(df: pd.DataFrame, default: str, col: Optional[str] = None) -> str:
    """Resolve a column name with case-insensitive fallback."""
    if col is not None and col in df.columns:
        return col
    if col is not None:
        for c in df.columns:
            if c.lower() == col.lower():
                return c
    if default in df.columns:
        return default
    for c in df.columns:
        if c.lower() == default.lower():
            return c
    raise KeyError(f"Column '{col or default}' not found in DataFrame")
