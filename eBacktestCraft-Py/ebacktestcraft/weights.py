"""Weight schemes — eBacktestCraft add_equal_weight / add_fixed_weight / add_norm_weight."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from equant.utils.panel import validate_panel


def equal_weight(
    df: pd.DataFrame,
    signal_col: str,
    weight_name: Optional[str] = None,
    zero_na: bool = True,
) -> pd.DataFrame:
    """Assign equal weights (1/n) to stocks with signal = 1.

    Stocks with signal != 1 receive weight 0.
    """
    validate_panel(df)
    if signal_col not in df.columns:
        raise ValueError(f"Signal column not found: {signal_col}")

    if weight_name is None:
        weight_name = f"weight_equal_{signal_col}"

    result = df.copy()
    signal_clean = result[signal_col].copy()

    if zero_na:
        signal_clean = signal_clean.fillna(0).replace([np.inf, -np.inf], 0)

    is_selected = (signal_clean == 1).astype(int)
    n_selected = result.groupby("date")[signal_col].transform(
        lambda x: (x.fillna(0) == 1).sum()
    )
    result[weight_name] = np.where(
        (n_selected > 0) & (is_selected == 1),
        1.0 / n_selected,
        0.0,
    )

    # Diagnostics
    daily_sum = result.groupby("date")[weight_name].sum()
    days_with_selection = (daily_sum > 0).sum()
    avg_selected = (daily_sum > 0).mean() * n_selected.mean() if len(daily_sum) > 0 else 0

    print(f" Generated equal weight column: {weight_name}")
    print(f" Total days: {len(daily_sum)}, days with selection: {days_with_selection}")

    return result


def fixed_weight(
    df: pd.DataFrame,
    signal_col: str,
    weights: dict[str, float],
    weight_name: Optional[str] = None,
) -> pd.DataFrame:
    """Assign fixed weights per asset based on signal.

    Parameters
    ----------
    weights : dict
        Mapping of asset ``code`` to weight (e.g., ``{"AAPL": 0.3, "MSFT": 0.5}``).
    """
    validate_panel(df)
    if signal_col not in df.columns:
        raise ValueError(f"Signal column not found: {signal_col}")

    if weight_name is None:
        weight_name = f"weight_fixed_{signal_col}"

    result = df.copy()
    result[weight_name] = 0.0

    for date in result["date"].unique():
        day_mask = result["date"] == date
        for code, weight in weights.items():
            code_mask = result["code"] == code
            sig = result.loc[day_mask & code_mask, signal_col]
            if len(sig) > 0 and sig.iloc[0] == 1:
                result.loc[day_mask & code_mask, weight_name] = weight

    return result


def norm_weight(
    df: pd.DataFrame,
    factor_col: str,
    signal_col: Optional[str] = None,
    weight_name: Optional[str] = None,
    long_only: bool = True,
) -> pd.DataFrame:
    """Assign weights proportional to factor values (normalized).

    Cross-sectionally: weight = factor / sum(|factor|) for selected assets.
    """
    validate_panel(df)
    if factor_col not in df.columns:
        raise ValueError(f"Factor column not found: {factor_col}")

    if weight_name is None:
        weight_name = f"weight_norm_{factor_col}"

    result = df.copy()

    def _norm_weights(group):
        factor = group[factor_col].copy()
        if long_only:
            factor = factor.clip(lower=0)
        s = factor.abs().sum()
        if s < 1e-15:
            return pd.Series(0.0, index=group.index)
        return factor / s

    result[weight_name] = result.groupby("date", group_keys=False).apply(
        _norm_weights, include_groups=False
    )

    # Apply signal filter if provided
    if signal_col is not None and signal_col in result.columns:
        result.loc[result[signal_col] != 1, weight_name] = 0.0

    return result
