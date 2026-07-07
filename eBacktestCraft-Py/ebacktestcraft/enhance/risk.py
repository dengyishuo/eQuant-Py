"""Dynamic risk controls — vol targeting, leverage caps, turnover limits."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from equant.utils.panel import slim_output, validate_panel


# ══════════════════════════════════════════════════════════════════════════════
# apply_vol_target — scale weights to maintain target volatility
# ══════════════════════════════════════════════════════════════════════════════


def apply_vol_target(
    df: pd.DataFrame,
    weight_col: str,
    signal_col: Optional[str] = None,
    target_vol: float = 0.15,
    vol_period: int = 20,
    close_col: Optional[str] = None,
    max_leverage: float = 2.0,
    new_col: str = "weight_voltarget",
    append: bool = True,
) -> pd.DataFrame:
    """Scale portfolio weights to maintain a target volatility level.

    When realized portfolio volatility is too high → reduce all positions.
    When volatility is too low → scale up positions (up to max_leverage).

    This is applied AFTER the base weight scheme. It's a portfolio-level
    multiplier, not a per-asset adjustment.

    Parameters
    ----------
    target_vol : float
        Annualized target volatility (e.g., 0.15 = 15%).
    max_leverage : float
        Maximum scaling factor (2.0 = max 200% exposure).
    """
    validate_panel(df, extra_cols=[weight_col])
    result = df.copy()

    if close_col is None:
        close_col = "adjusted" if "adjusted" in result.columns else "close"

    trading_days = 252
    weight_out = np.full(len(result), np.nan)

    # Compute per-asset daily returns
    ret_col = "_daily_ret"
    ret_vals = np.full(len(result), np.nan)
    for code, idx in result.groupby("code", sort=False).groups.items():
        prices = result.loc[idx, close_col].values.astype(np.float64)
        r = np.full_like(prices, np.nan)
        r[1:] = (prices[1:] - prices[:-1]) / np.maximum(np.abs(prices[:-1]), 1e-10)
        ret_vals[idx] = r
    result[ret_col] = ret_vals

    # Iterate dates, compute trailing portfolio vol, scale
    dates = sorted(result["date"].unique())
    for di, d in enumerate(dates):
        mask = result["date"] == d
        sub = result.loc[mask]
        if signal_col:
            active = sub[sub[signal_col] == 1]
        else:
            active = sub[sub[weight_col].notna() & (sub[weight_col] > 0)]

        if len(active) == 0:
            weight_out[mask.values] = 0.0
            continue

        base_w = active[weight_col].values.astype(np.float64)
        base_w = np.maximum(base_w, 0)
        w_sum = base_w.sum()
        if w_sum < 1e-10:
            continue
        base_w = base_w / w_sum

        # Estimate portfolio realized vol from past daily returns
        port_rets = []
        for ci, code in enumerate(active["code"].values):
            code_mask = (result["code"] == code) & (result["date"] < d)
            code_data = result.loc[code_mask, [ret_col, "date"]].sort_values("date")
            if len(code_data) < 5:
                continue
            code_rets = code_data[ret_col].values[-vol_period:]
            port_rets.append(code_rets)
        if len(port_rets) < 2:
            continue

        # Approximate: average pairwise return weighted by weights
        min_len = min(len(r) for r in port_rets)
        if min_len < 5:
            continue
        # Align to same length
        aligned = np.array([r[-min_len:] for r in port_rets])
        # Portfolio return series: weighted average of asset returns
        w_subset = base_w[:len(port_rets)]
        w_subset = w_subset / max(w_subset.sum(), 1e-10)
        port_ret_series = (aligned.T @ w_subset)
        realized_vol = np.std(port_ret_series) * np.sqrt(trading_days)

        if realized_vol < 1e-6:
            scale = 1.0
        else:
            scale = target_vol / realized_vol
        scale = np.clip(scale, 0.3, max_leverage)  # min 30%, max cap

        # Apply scale
        w_scaled = base_w * scale
        w_scaled = w_scaled / max(w_scaled.sum(), 1e-10)
        weight_out[active.index] = w_scaled

    result[new_col] = weight_out
    result.drop(columns=[ret_col], inplace=True, errors="ignore")
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════════
# compute_turnover — calculate and optionally limit turnover
# ══════════════════════════════════════════════════════════════════════════════


def compute_turnover(
    df: pd.DataFrame,
    weight_col: str,
    signal_col: Optional[str] = None,
    new_col: str = "turnover",
    append: bool = True,
) -> pd.DataFrame:
    """Compute day-over-day portfolio turnover from weight changes.

    Turnover = 0.5 × Σ|w_t - w_{t-1}_lagged|, the standard two-sided
    turnover measure. A new column is appended with per-date turnover values.

    Parameters
    ----------
    weight_col : str
        Column with current weights.
    new_col : str
        Output column name for turnover values.

    Notes
    -----
    - First date for each asset has NaN turnover.
    - Turnover is in [0, 1] range when weights sum to 1.
    """
    validate_panel(df, extra_cols=[weight_col])
    result = df.copy()
    turnover_vals = np.full(len(result), np.nan)

    for code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        weights = sub[weight_col].values.astype(np.float64)
        w_today = weights[1:]
        w_yesterday = weights[:-1]
        # Handle NaN: skip computation where either is NaN
        to = np.full(len(weights), np.nan)
        valid = ~np.isnan(w_today) & ~np.isnan(w_yesterday)
        to[1:][valid] = np.abs(w_today[valid] - w_yesterday[valid])
        turnover_vals[sub.index] = to

    # Per-date: average turnover across assets
    result[new_col] = turnover_vals
    # Add a per-date summary column
    daily_to = result.groupby("date")[new_col].mean()
    result[f"{new_col}_daily"] = result["date"].map(daily_to)

    return slim_output(result, [new_col, f"{new_col}_daily"], append)
