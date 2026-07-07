"""Panel data validation and manipulation utilities.

All quantkit functions operate on long-format panel DataFrames with
the convention: one row per asset-date observation, with required
identifier columns ``date``, ``code``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

# Required identifier columns in every panel DataFrame
_ID_COLS = {"date", "code"}


def required_id_cols() -> frozenset:
    """Return the required identifier column names."""
    return frozenset(_ID_COLS)


def validate_panel(df: pd.DataFrame, extra_cols: Optional[Sequence[str]] = None) -> None:
    """Validate that *df* conforms to the long-format panel convention.

    Raises ``ValueError`` if required identifier columns or any *extra_cols*
    are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"'df' must be a pandas DataFrame, got {type(df).__name__}")

    missing = _ID_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Panel DataFrame missing required identifier columns: "
            f"{', '.join(sorted(missing))}"
        )

    if extra_cols:
        missing_extra = set(extra_cols) - set(df.columns)
        if missing_extra:
            raise ValueError(
                f"Panel DataFrame missing required data columns: "
                f"{', '.join(sorted(missing_extra))}"
            )


def sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Sort panel DataFrame by date then code (chronological within each cross-section).

    Returns a new DataFrame; does not mutate the input.
    """
    return df.sort_values(["date", "code"]).reset_index(drop=True)


def ensure_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Validate that *cols* exist in *df* and return *df* unchanged.

    Convenience wrapper around :func:`validate_panel` for mid-pipeline checks.
    """
    validate_panel(df, extra_cols=cols)
    return df


def slim_output(
    df: pd.DataFrame,
    new_col: str | Sequence[str],
    append: bool,
) -> pd.DataFrame:
    """Return slimmed or full output depending on *append*.

    When ``append=False``, returns only ``[date, code, name, *new_cols]``.
    When ``append=True``, returns the full DataFrame unchanged.
    """
    if append:
        return df

    if isinstance(new_col, str):
        new_col = [new_col]

    keep = ["date", "code", "name"]
    # Only keep columns that actually exist (name is optional in some contexts)
    keep = [c for c in keep if c in df.columns]
    keep += list(new_col)
    return df[keep]
