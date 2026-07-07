"""Factor ranking and quantile utilities — eFactorCraft equivalent."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from equant.utils.panel import slim_output, validate_panel


def quantile_rank(
    df: pd.DataFrame,
    factor_cols: Union[str, Sequence[str]],
    type: str = "cross",
    n: int = 60,
    by: str = "date",
    ties_method: str = "average",
    new_col: str = "qrank",
    append: bool = True,
) -> pd.DataFrame:
    """Quantile (percentile) rank of factor values.

    Parameters
    ----------
    factor_cols : str or sequence
        Factor columns to rank.
    type : str
        ``"cross"`` = cross-sectional rank per date.
        ``"time"`` = time-series rank per asset within rolling window.
    n : int
        Rolling window length for ``type="time"``.
    by : str
        Grouping column for cross-sectional rank. Default ``"date"``.
    ties_method : str
        Passed to pandas ``rank()``.
    new_col : str
        Prefix for output columns.
    """
    validate_panel(df)
    cols = [factor_cols] if isinstance(factor_cols, str) else list(factor_cols)
    result = df.copy()

    for col in cols:
        out_name = f"{new_col}_{col}"

        if type == "cross":
            result[out_name] = result.groupby(by)[col].rank(pct=True, method=ties_method)
        elif type == "time":
            result[out_name] = np.nan
            for _code, idx in result.groupby("code", sort=False).groups.items():
                sub = result.loc[idx].sort_values("date")

                def _pct_rank(w):
                    w = w.dropna()
                    if len(w) < 2:
                        return np.nan
                    r = w.rank(pct=True, method=ties_method)
                    return r.iloc[-1]

                rolled = sub[col].rolling(window=n, min_periods=2)
                result.loc[sub.index, out_name] = rolled.apply(_pct_rank, raw=False).values
        else:
            raise ValueError(f"Unknown type: {type}. Use 'cross' or 'time'.")

    out_cols = [f"{new_col}_{c}" for c in cols]
    return slim_output(result, out_cols, append)


def quantile_flag(
    df: pd.DataFrame,
    factor_col: str,
    n_groups: int = 10,
    by: str = "date",
    new_col: str = "quantile_group",
    append: bool = True,
) -> pd.DataFrame:
    """Assign assets to quantile groups per cross-section.

    Returns integer group labels 1..n_groups (1 = lowest, n_groups = highest).
    """
    validate_panel(df)
    result = df.copy()

    def _qcut(g):
        try:
            return pd.qcut(g, n_groups, labels=False, duplicates="drop") + 1
        except ValueError:
            return pd.Series(np.nan, index=g.index)

    result[new_col] = result.groupby(by)[factor_col].transform(_qcut)

    return slim_output(result, new_col, append)


def consecutive_days(
    df: pd.DataFrame,
    flag_col: str,
    new_col: str = "consecutive_days",
    append: bool = True,
) -> pd.DataFrame:
    """Count consecutive days a condition holds.

    Counter resets when *flag_col* changes or is NaN.
    """
    validate_panel(df)
    result = df.copy()
    result[new_col] = 0

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        flag = sub[flag_col].values
        count = np.zeros(len(flag), dtype=int)
        streak = 1
        for i in range(1, len(flag)):
            if np.isnan(flag[i]) or flag[i] != flag[i - 1]:
                streak = 1
            else:
                streak += 1
            count[i] = streak
        result.loc[sub.index, new_col] = count

    return slim_output(result, new_col, append)
