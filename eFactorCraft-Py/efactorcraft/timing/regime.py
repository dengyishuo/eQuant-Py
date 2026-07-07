"""Factor timing — regime detection and dynamic factor adjustment.

Market regime classification and rule-based factor exposure adjustment.
All DataFrame-returning functions follow the standard validate/copy/slim pattern.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from equant.utils.panel import slim_output, validate_panel


def _resolve_close(df: pd.DataFrame, close_col: Optional[str] = None) -> str:
    """Resolve close column name with case-insensitive fallback."""
    if close_col is not None:
        if close_col in df.columns:
            return close_col
        lower = df.columns.str.lower()
        if close_col.lower() in lower.values:
            return df.columns[lower == close_col.lower()][0]
    for candidate in ["adjusted", "close", "Adj Close", "Close"]:
        if candidate in df.columns:
            return candidate
        lower = df.columns.str.lower()
        if candidate.lower() in lower.values:
            return df.columns[lower == candidate.lower()][0]
    return "close"


# ══════════════════════════════════════════════════════════════════════════════
# regime_detect
# ══════════════════════════════════════════════════════════════════════════════


def regime_detect(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    ma_period: int = 60,
    vol_period: int = 20,
    vol_threshold: float = 1.5,
    new_col: str = "regime",
    append: bool = True,
) -> pd.DataFrame:
    """Classify each asset-date into a market regime.

    Regimes:
    - ``"bull"``: close > MA(close) AND volatility < vol_threshold * mean_vol
    - ``"bear"``: close < MA(close) AND volatility > vol_threshold * mean_vol
    - ``"sideways"``: everything else
    - ``"unknown"``: insufficient data (early periods with NaN)

    Parameters
    ----------
    close_col : str, optional
        Price column. Defaults to "adjusted", falls back to "close".
    ma_period : int
        Moving average window for trend detection.
    vol_period : int
        Volatility window for regime classification.
    vol_threshold : float
        Multiplier on long-term mean volatility.
    """
    validate_panel(df)
    result = df.copy()
    close = _resolve_close(result, close_col)
    regime_vals = np.full(len(result), "unknown", dtype=object)

    for code, idx in result.groupby("code", sort=False).groups.items():
        prices = result.loc[idx, close].values.astype(np.float64)
        # Returns
        rets = np.full_like(prices, np.nan)
        rets[1:] = (prices[1:] - prices[:-1]) / np.maximum(np.abs(prices[:-1]), 1e-10)

        # Rolling MA
        ma = pd.Series(prices).rolling(ma_period, min_periods=ma_period).mean().values
        # Current volatility
        vol = pd.Series(rets).rolling(vol_period, min_periods=5).std().values
        # Long-term volatility
        long_vol = pd.Series(rets).rolling(vol_period * 5, min_periods=vol_period).std().values

        for i in range(len(prices)):
            if np.isnan(ma[i]) or np.isnan(vol[i]) or np.isnan(long_vol[i]):
                continue  # stays "unknown"
            above_ma = prices[i] > ma[i]
            below_ma = prices[i] < ma[i]
            low_vol = vol[i] < vol_threshold * max(long_vol[i], 1e-10)
            high_vol = vol[i] > vol_threshold * max(long_vol[i], 1e-10)

            if above_ma and low_vol:
                regime_vals[idx[i]] = "bull"
            elif below_ma and high_vol:
                regime_vals[idx[i]] = "bear"
            else:
                regime_vals[idx[i]] = "sideways"

    result[new_col] = regime_vals
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════════
# trend_filter
# ══════════════════════════════════════════════════════════════════════════════


def trend_filter(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    ma_period: int = 60,
    new_col: str = "trend",
    append: bool = True,
) -> pd.DataFrame:
    """Continuous trend signal: close / MA(close).

    Values > 1 = uptrend, < 1 = downtrend. Can be used as a factor multiplier.
    """
    validate_panel(df)
    result = df.copy()
    close = _resolve_close(result, close_col)
    trend_vals = np.full(len(result), np.nan)

    for code, idx in result.groupby("code", sort=False).groups.items():
        prices = result.loc[idx, close].values.astype(np.float64)
        ma = pd.Series(prices).rolling(ma_period, min_periods=ma_period).mean().values
        trend_vals[idx] = prices / np.maximum(ma, 1e-10)

    result[new_col] = trend_vals
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════════
# vol_filter
# ══════════════════════════════════════════════════════════════════════════════


def vol_filter(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    vol_period: int = 20,
    lookback: int = 252,
    new_col: str = "vol_regime",
    append: bool = True,
) -> pd.DataFrame:
    """Continuous volatility regime: long_term_vol / current_vol.

    Values > 1 = current vol is below historical average (low vol regime).
    Values < 1 = current vol is elevated (high vol regime).
    """
    validate_panel(df)
    result = df.copy()
    close = _resolve_close(result, close_col)
    vol_ratio = np.full(len(result), np.nan)

    for code, idx in result.groupby("code", sort=False).groups.items():
        prices = result.loc[idx, close].values.astype(np.float64)
        rets = np.full_like(prices, np.nan)
        rets[1:] = (prices[1:] - prices[:-1]) / np.maximum(np.abs(prices[:-1]), 1e-10)
        cur_vol = pd.Series(rets).rolling(vol_period, min_periods=5).std().values
        long_vol = pd.Series(rets).rolling(lookback, min_periods=vol_period).std().values
        vol_ratio[idx] = long_vol / np.maximum(cur_vol, 1e-10)

    result[new_col] = vol_ratio
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════════
# timing_weight
# ══════════════════════════════════════════════════════════════════════════════


def timing_weight(
    df: pd.DataFrame,
    factor_col: str,
    regime_col: str = "regime",
    weights: Optional[dict[str, float]] = None,
    new_col_prefix: str = "timed",
    append: bool = True,
) -> pd.DataFrame:
    """Apply regime-dependent weight multiplier to a factor.

    Parameters
    ----------
    weights : dict, optional
        Mapping from regime to weight multiplier.
        Default: ``{"bull": 1.0, "bear": 0.0, "sideways": 0.5, "unknown": 0.5}``
    new_col_prefix : str
        Output column named ``{new_col_prefix}_{factor_col}``.
    """
    validate_panel(df, extra_cols=[factor_col])
    if weights is None:
        weights = {"bull": 1.0, "bear": 0.0, "sideways": 0.5, "unknown": 0.5}

    result = df.copy()
    out_col = f"{new_col_prefix}_{factor_col}"
    result[out_col] = np.nan

    for idx in result.index:
        regime = str(result.loc[idx, regime_col]) if regime_col in result.columns else "unknown"
        mult = weights.get(regime, 0.5)
        result.loc[idx, out_col] = result.loc[idx, factor_col] * mult

    return slim_output(result, out_col, append)


# ══════════════════════════════════════════════════════════════════════════════
# adaptive_composite
# ══════════════════════════════════════════════════════════════════════════════


def adaptive_composite(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    regime_col: str = "regime",
    regime_factor_map: Optional[dict[str, list[str]]] = None,
    forward_col: Optional[str] = None,
    window: int = 60,
    new_col: str = "composite_adaptive",
    append: bool = True,
) -> pd.DataFrame:
    """Regime-adaptive factor composite.

    Parameters
    ----------
    regime_factor_map : dict, optional
        Maps regime to subset of ``factor_cols``. E.g.
        ``{"bull": ["mom_5"], "bear": ["value", "size"], "sideways": ["rps_60"]}``.
        If None, all factors are used in all regimes, weighted by rolling-IC
        within each regime.
    """
    validate_panel(df, extra_cols=list(factor_cols))
    result = df.copy()
    composite = np.full(len(result), np.nan)

    if regime_factor_map is not None:
        # Use specified factor subsets per regime
        for regime, cols in regime_factor_map.items():
            mask = result[regime_col] == regime
            if not mask.any():
                continue
            # Validate that all cols exist
            missing = [c for c in cols if c not in result.columns]
            if missing:
                raise ValueError(f"Columns {missing} not found for regime '{regime}'")
            # Z-score equal-weight composite within regime
            sub = result.loc[mask, cols].copy()
            for c in cols:
                grp_mean = sub[c].mean()
                grp_std = sub[c].std()
                if grp_std and grp_std > 0:
                    sub[c] = (sub[c] - grp_mean) / grp_std
                else:
                    sub[c] = 0.0
            composite[mask.values] = sub.mean(axis=1, skipna=True).values
    else:
        # Use all factors in all regimes, IC-weighted
        if forward_col is None:
            fwd_candidates = [c for c in result.columns if c.startswith("forward_")]
            if not fwd_candidates:
                raise ValueError("No forward_* column found. Run eng.add_next_return() first.")
            forward_col = fwd_candidates[0]

        dates = sorted(result["date"].unique())
        for regime in result[regime_col].unique():
            regime_mask = result[regime_col] == regime
            if not regime_mask.any():
                continue
            regime_dates = sorted(result.loc[regime_mask, "date"].unique())
            # Compute per-factor IC within this regime
            ic_dict = {}
            for fc in factor_cols:
                ic_list = []
                for d in dates:
                    sub = result[(result["date"] == d) & (result[regime_col] == regime)]
                    if len(sub) < 5:
                        continue
                    valid = sub[[fc, forward_col]].dropna()
                    if len(valid) >= 5:
                        ic_list.append((d, valid[fc].corr(valid[forward_col])))
                ic_dict[fc] = pd.Series(
                    [v for _, v in ic_list],
                    index=pd.DatetimeIndex([d for d, _ in ic_list]),
                )

            for di, d in enumerate(regime_dates):
                mask = (result["date"] == d) & regime_mask
                if not mask.any():
                    continue
                raw_w = []
                for fc in factor_cols:
                    ic_vals = ic_dict[fc]
                    trailing = ic_vals[ic_vals.index <= d].iloc[-window:] if len(ic_vals) > 0 else pd.Series(dtype=float)
                    if len(trailing) < 5:
                        raw_w.append(0.0)
                    else:
                        raw_w.append(max(trailing.mean(), 0.0))
                raw_w = np.array(raw_w)
                abs_sum = np.sum(np.abs(raw_w))
                if abs_sum < 1e-10:
                    continue
                weights = raw_w / abs_sum
                sub = result.loc[mask, factor_cols]
                sub_std = sub.copy()
                for fc in factor_cols:
                    m = sub[fc].mean()
                    s = sub[fc].std()
                    if s and s > 0:
                        sub_std[fc] = (sub[fc] - m) / s
                    else:
                        sub_std[fc] = 0.0
                comp = np.zeros(len(sub_std))
                for fi, fc in enumerate(factor_cols):
                    comp += sub_std[fc].values * weights[fi]
                composite[mask.values] = comp

    result[new_col] = composite
    return slim_output(result, new_col, append)
