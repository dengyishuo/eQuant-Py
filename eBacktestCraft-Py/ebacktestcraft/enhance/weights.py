"""Enhanced weight schemes — risk-aware position sizing.

Each function takes a long-format panel and appends a weight column.
All follow: validate_panel → copy → compute → slim_output.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from equant.utils.panel import slim_output, validate_panel


# ══════════════════════════════════════════════════════════════════════════════
# vol_parity_weight — inverse volatility weighting
# ══════════════════════════════════════════════════════════════════════════════


def vol_parity_weight(
    df: pd.DataFrame,
    signal_col: str,
    vol_col: Optional[str] = None,
    vol_period: int = 20,
    close_col: Optional[str] = None,
    new_col: str = "weight_vp",
    append: bool = True,
) -> pd.DataFrame:
    """Inverse-volatility parity weight: weight ∝ 1/σ.

    Each selected asset gets weight proportional to 1/volatility.
    High-volatility stocks get smaller weights, reducing drawdowns.

    Parameters
    ----------
    signal_col : str
        0/1 signal column. Only assets with signal==1 receive weight.
    vol_col : str, optional
        Pre-computed volatility column. If None, computes from close_col.
    vol_period : int
        Rolling volatility window if vol_col is not provided.
    close_col : str, optional
        Price column for volatility calculation. Defaults to "adjusted".
    new_col : str
        Output weight column name.
    """
    validate_panel(df, extra_cols=[signal_col])
    result = df.copy()

    # Resolve volatility column
    if vol_col is None:
        if close_col is None:
            close_col = "adjusted" if "adjusted" in result.columns else "close"
        vol_col = f"_vol_{vol_period}"
        # Compute rolling realized volatility per asset
        vol_vals = np.full(len(result), np.nan)
        for code, idx in result.groupby("code", sort=False).groups.items():
            prices = result.loc[idx, close_col].values.astype(np.float64)
            rets = np.full_like(prices, np.nan)
            rets[1:] = (prices[1:] - prices[:-1]) / np.maximum(np.abs(prices[:-1]), 1e-10)
            rv = pd.Series(rets).rolling(vol_period, min_periods=5).std().values
            vol_vals[idx] = rv
        result[vol_col] = vol_vals

    weight_vals = np.full(len(result), 0.0)
    for date, idx in result.groupby("date", sort=False).groups.items():
        sub = result.loc[idx]
        selected = sub[sub[signal_col] == 1]
        if len(selected) == 0:
            continue
        vols = selected[vol_col].values.astype(np.float64)
        # Use inverse vol, capped to avoid extreme weights
        inv_vol = 1.0 / np.maximum(vols, 1e-4)
        # Cap each weight at 5x the average
        avg_w = inv_vol.mean()
        inv_vol = np.clip(inv_vol, 0, avg_w * 5)
        w = inv_vol / max(inv_vol.sum(), 1e-10)
        weight_vals[selected.index] = w

    result[new_col] = weight_vals
    # Clean up temp column
    if vol_col.startswith("_vol_"):
        result.drop(columns=[vol_col], inplace=True, errors="ignore")
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════════
# target_vol_weight — scale positions to hit a volatility target
# ══════════════════════════════════════════════════════════════════════════════


def target_vol_weight(
    df: pd.DataFrame,
    signal_col: str,
    target_vol: float = 0.15,
    vol_period: int = 20,
    close_col: Optional[str] = None,
    base_weight_col: Optional[str] = None,
    new_col: str = "weight_tv",
    append: bool = True,
) -> pd.DataFrame:
    """Scale weights so the portfolio's realized volatility hits target_vol.

    weight_final = weight_base × target_vol / portfolio_realized_vol

    Parameters
    ----------
    target_vol : float
        Annualized target volatility (e.g., 0.15 = 15%).
    base_weight_col : str, optional
        Column with raw weights to scale. If None, uses equal weight.
    """
    validate_panel(df, extra_cols=[signal_col])
    result = df.copy()

    if close_col is None:
        close_col = "adjusted" if "adjusted" in result.columns else "close"

    # Compute per-asset rolling volatility
    vol_col = f"_rv_{vol_period}"
    vol_vals = np.full(len(result), np.nan)
    for code, idx in result.groupby("code", sort=False).groups.items():
        prices = result.loc[idx, close_col].values.astype(np.float64)
        rets = np.full_like(prices, np.nan)
        rets[1:] = (prices[1:] - prices[:-1]) / np.maximum(np.abs(prices[:-1]), 1e-10)
        vol_vals[idx] = pd.Series(rets).rolling(vol_period, min_periods=5).std().values
    result[vol_col] = vol_vals

    weight_vals = np.full(len(result), 0.0)
    trading_days = 252

    for date, idx in result.groupby("date", sort=False).groups.items():
        sub = result.loc[idx]
        selected = sub[sub[signal_col] == 1]
        if len(selected) == 0:
            continue

        # Get base weights
        if base_weight_col and base_weight_col in result.columns:
            base_w = selected[base_weight_col].values.astype(np.float64)
            base_w = np.maximum(base_w, 0)
            base_w = base_w / max(base_w.sum(), 1e-10)
        else:
            base_w = np.full(len(selected), 1.0 / len(selected))

        # Estimate portfolio vol from constituent vols and base weights
        vols = selected[vol_col].values.astype(np.float64)
        port_vol = np.sqrt(np.sum((base_w * vols)**2)) * np.sqrt(trading_days)

        if port_vol < 1e-6:
            continue

        # Scale factor
        scale = min(target_vol / port_vol, 2.0)  # cap at 2x to avoid extreme leverage
        w = base_w * scale
        w = w / max(w.sum(), 1e-10)
        weight_vals[selected.index] = w

    result[new_col] = weight_vals
    result.drop(columns=[vol_col], inplace=True, errors="ignore")
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════════
# erp_weight — Expected Return Per unit of risk (factor / volatility)
# ══════════════════════════════════════════════════════════════════════════════


def erp_weight(
    df: pd.DataFrame,
    factor_col: str,
    signal_col: Optional[str] = None,
    vol_period: int = 20,
    close_col: Optional[str] = None,
    new_col: str = "weight_erp",
    append: bool = True,
) -> pd.DataFrame:
    """Expected Return Per-risk weight: weight ∝ factor / volatility.

    Combines factor strength with risk adjustment. Cross-sectionally
    normalized so weights sum to 1 per date.

    Parameters
    ----------
    factor_col : str
        Factor/alpha column (e.g., composite score).
    signal_col : str, optional
        Signal filter column. Only assets with signal==1 receive weight.
        If None, all assets receive weight.
    """
    validate_panel(df, extra_cols=[factor_col])
    result = df.copy()

    if close_col is None:
        close_col = "adjusted" if "adjusted" in result.columns else "close"

    # Compute rolling volatility
    vol_col = f"_rv_{vol_period}"
    vol_vals = np.full(len(result), np.nan)
    for code, idx in result.groupby("code", sort=False).groups.items():
        prices = result.loc[idx, close_col].values.astype(np.float64)
        rets = np.full_like(prices, np.nan)
        rets[1:] = (prices[1:] - prices[:-1]) / np.maximum(np.abs(prices[:-1]), 1e-10)
        vol_vals[idx] = pd.Series(rets).rolling(vol_period, min_periods=5).std().values
    result[vol_col] = vol_vals

    weight_vals = np.full(len(result), 0.0)

    for date, idx in result.groupby("date", sort=False).groups.items():
        sub = result.loc[idx]
        if signal_col:
            sub = sub[sub[signal_col] == 1]
        if len(sub) == 0:
            continue

        factor_vals = sub[factor_col].values.astype(np.float64)
        vols = sub[vol_col].values.astype(np.float64)

        # erp = factor / vol (factor must be standardized first)
        # Standardize factor cross-sectionally
        f_mean = np.nanmean(factor_vals)
        f_std = np.nanstd(factor_vals)
        if f_std < 1e-10:
            continue
        f_std = (factor_vals - f_mean) / f_std

        erp = f_std / np.maximum(vols, 1e-4)
        # Only go long (positive factor = positive weight)
        erp = np.maximum(erp, 0)
        abs_sum = np.sum(np.abs(erp))
        if abs_sum < 1e-10:
            continue
        weight_vals[sub.index] = erp / abs_sum

    result[new_col] = weight_vals
    result.drop(columns=[vol_col], inplace=True, errors="ignore")
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════════
# confidence_weight — signal strength × conviction
# ══════════════════════════════════════════════════════════════════════════════


def confidence_weight(
    df: pd.DataFrame,
    factor_col: str,
    signal_col: Optional[str] = None,
    min_weight: float = 0.0,
    max_weight: float = 0.40,
    new_col: str = "weight_cf",
    append: bool = True,
) -> pd.DataFrame:
    """Weight proportional to absolute factor strength.

    Stronger signals → larger positions. Zero or negative factor → no position.
    Cross-sectionally the weights approximate: weight ∝ max(factor, 0).

    Parameters
    ----------
    min_weight : float
        Minimum weight per asset (enforced after normalization).
    max_weight : float
        Maximum weight per asset (single-position cap).
    """
    validate_panel(df, extra_cols=[factor_col])
    result = df.copy()

    weight_vals = np.full(len(result), 0.0)

    for date, idx in result.groupby("date", sort=False).groups.items():
        sub = result.loc[idx]
        if signal_col:
            sub = sub[sub[signal_col] == 1]
        if len(sub) == 0:
            continue

        factor_vals = sub[factor_col].values.astype(np.float64)

        # Standardize cross-sectionally
        f_mean = np.nanmean(factor_vals)
        f_std = np.nanstd(factor_vals)
        if f_std < 1e-10:
            continue
        f_z = (factor_vals - f_mean) / f_std

        # Softmax-style: only go long on positive z-scores
        raw = np.maximum(f_z, 0)
        # Add small baseline so all selected get some weight
        raw = raw + 0.1
        total = raw.sum()
        if total < 1e-10:
            continue
        w = raw / total
        # Normalize first, then clip. If sum>1 after clip, scale down.
        w = raw / total
        w = np.clip(w, min_weight, max_weight)
        s = w.sum()
        if s > 1.0:
            w = w / s  # scale down proportionally if caps are exceeded
        weight_vals[sub.index] = w

    result[new_col] = weight_vals
    return slim_output(result, new_col, append)
