"""
Shared scaffolding for all add_alphaXXX functions.

Every public function follows this contract:

    def add_alphaXXX(
        mkt_data: pd.DataFrame,
        close_col: str = "close",
        ...
        append: bool = True,
    ) -> pd.DataFrame:

Input
-----
mkt_data : pd.DataFrame
    Long-format panel with at minimum columns:
      date (datetime-like), code (str), name (str)
    plus whatever price/volume/fundamental columns the alpha needs.

Output
------
pd.DataFrame
    Same rows as input (order preserved).  When append=True the alpha
    column is appended; when False only date/code/name + alpha column.
"""

from __future__ import annotations
import pandas as pd


IDENTITY_COLS = ["date", "code", "name"]


def _validate(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"mkt_data is missing required columns: {missing}")


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def _finish(
    df: pd.DataFrame,
    original_index,
    alpha_col: str,
    append: bool,
    tmp_cols: list[str],
) -> pd.DataFrame:
    """Drop temp columns, optionally slim to identity+alpha, restore index."""
    df = df.drop(columns=[c for c in tmp_cols if c in df.columns], errors="ignore")
    if not append:
        df = df[IDENTITY_COLS + [alpha_col]].copy()
    df.index = original_index
    return df
