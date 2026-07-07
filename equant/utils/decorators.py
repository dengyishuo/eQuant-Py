"""Decorators for equant functions.

Provides common patterns: panel validation, copy-safety, and
append/output formatting.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

import pandas as pd

from equant.utils.panel import slim_output, sort_panel, validate_panel


def panel_aware(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    """Decorate a function that takes a panel DataFrame as first argument.

    Automatically:
    1. Copies the input DataFrame (no side-effects on caller's data)
    2. Validates the panel has required ``date`` and ``code`` columns
    3. Sorts the panel before processing
    """

    @functools.wraps(func)
    def wrapper(df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        df = df.copy()
        validate_panel(df)
        df = sort_panel(df)
        return func(df, *args, **kwargs)

    return wrapper


def copy_safe(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    """Ensure the first argument (DataFrame) is copied before mutation.

    Lighter than ``panel_aware`` — no validation or sorting, just a copy.
    """

    @functools.wraps(func)
    def wrapper(df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return func(df.copy(), *args, **kwargs)

    return wrapper


def with_append_output(
    new_col_param: str = "new_col",
) -> Callable[..., Callable[..., pd.DataFrame]]:
    """Factory: decorate a function to auto-handle ``append` / ``output`` params.

    The decorated function should accept ``append: bool = True`` and
    ``output: str = "dataframe"`` as keyword arguments. This decorator
    handles the slim-output logic after the function runs.

    Parameters
    ----------
    new_col_param : str
        Name of the keyword argument that holds the new column name(s).
    """

    def decorator(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> pd.DataFrame:
            append = kwargs.pop("append", True)
            output = kwargs.pop("output", "dataframe")

            # Extract new_col from kwargs before calling function
            new_col = kwargs.get(new_col_param, None)

            result = func(*args, **kwargs)

            if not append and new_col is not None:
                result = slim_output(result, new_col, append=False)

            if output == "tibble":
                # tibble-compatible: ensure modern pandas display
                pass

            return result

        return wrapper

    return decorator
