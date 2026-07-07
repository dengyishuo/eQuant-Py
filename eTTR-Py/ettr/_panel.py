"""Panel-data helpers for indicators.

Provides the split-apply-combine pattern that wraps eTTR's
``lapply(unique(code), ...)`` approach for applying indicator
functions to each asset independently in a long-format panel.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from equant.utils.panel import slim_output, sort_panel, validate_panel


def apply_by_code(
    df: pd.DataFrame,
    func: Callable[..., pd.DataFrame],
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """Apply *func* to each asset group independently.

    Equivalent to eTTR's::

        result <- do.call(rbind, lapply(unique(code), function(cd) {
            sub <- mkt_data[mkt_data$code == cd, ]
            func(sub, ...)
        }))
    """
    validate_panel(df)
    df = sort_panel(df)

    results: list[pd.DataFrame] = []
    for _code, group in df.groupby("code", sort=False):
        result = func(group, *args, **kwargs)
        results.append(result)

    if not results:
        return df.iloc[:0]

    out = pd.concat(results, ignore_index=True)
    return sort_panel(out)


def _resolve_col(df: pd.DataFrame, default: str, col: str | None = None) -> str:
    """Resolve a column name with case-insensitive fallback."""
    if col is not None and col in df.columns:
        return col
    if col is not None:
        # Try case-insensitive match
        for c in df.columns:
            if c.lower() == col.lower():
                return c
    if default in df.columns:
        return default
    for c in df.columns:
        if c.lower() == default.lower():
            return c
    raise KeyError(f"Column '{col or default}' not found in DataFrame")
