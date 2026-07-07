"""Factor synthesis — combine multiple factor columns into composite scores.

Each function accepts a long-format panel and appends a single composite
score column. All follow the standard pattern: validate_panel → copy →
compute → slim_output.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from equant.utils.panel import slim_output, validate_panel


def _std_factors(df: pd.DataFrame, factor_cols: Sequence[str], by: str = "date") -> pd.DataFrame:
    """Cross-sectionally z-score standardize factors per grouping column."""
    result = df.copy()
    for fc in factor_cols:
        grp = df.groupby(by)[fc]
        result[f"_std_{fc}"] = (df[fc] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    return result


def _rank_factors(df: pd.DataFrame, factor_cols: Sequence[str], by: str = "date") -> pd.DataFrame:
    """Cross-sectionally percentile-rank factors per grouping column."""
    result = df.copy()
    for fc in factor_cols:
        result[f"_rank_{fc}"] = df.groupby(by)[fc].transform(
            lambda x: x.rank(pct=True, method="average")
        )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# equal_weighted_composite
# ══════════════════════════════════════════════════════════════════════════════


def equal_weighted_composite(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    by: str = "date",
    new_col: str = "composite_ew",
    append: bool = True,
) -> pd.DataFrame:
    """Equal-weighted composite of z-scored factors.

    Parameters
    ----------
    df : DataFrame
        Long-format panel.
    factor_cols : sequence of str
        Factor column names to combine.
    by : str
        Cross-sectional grouping column (default "date").
    new_col : str
        Output column name.
    append : bool
        If False, return only id columns + new column.
    """
    validate_panel(df, extra_cols=factor_cols)
    result = _std_factors(df, factor_cols, by)
    std_cols = [f"_std_{fc}" for fc in factor_cols]
    result[new_col] = result[std_cols].mean(axis=1, skipna=True)
    result.drop(columns=std_cols, inplace=True)
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════════
# rank_weighted_composite
# ══════════════════════════════════════════════════════════════════════════════


def rank_weighted_composite(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    by: str = "date",
    new_col: str = "composite_rw",
    append: bool = True,
) -> pd.DataFrame:
    """Composite averaging cross-sectional percentile ranks of factors.

    Robust to outliers — no standardization needed.
    """
    validate_panel(df, extra_cols=factor_cols)
    result = _rank_factors(df, factor_cols, by)
    rank_cols = [f"_rank_{fc}" for fc in factor_cols]
    result[new_col] = result[rank_cols].mean(axis=1, skipna=True)
    result.drop(columns=rank_cols, inplace=True)
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════════
# ic_weighted_composite
# ══════════════════════════════════════════════════════════════════════════════


def ic_weighted_composite(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    forward_col: Optional[str] = None,
    window: int = 60,
    by: str = "date",
    new_col: str = "composite_icw",
    append: bool = True,
) -> pd.DataFrame:
    """Weighted composite using trailing Information Coefficient as weights.

    For each date, the weight of each factor is its rolling mean IC over
    the trailing ``window`` days. Factors are z-scored before weighting.

    Parameters
    ----------
    forward_col : str, optional
        Forward return column.  Auto-detects first ``forward_*`` column if None.
    window : int
        Rolling window for trailing IC calculation.
    """
    validate_panel(df, extra_cols=factor_cols)
    # Resolve forward column
    if forward_col is None:
        fwd_candidates = [c for c in df.columns if c.startswith("forward_")]
        if not fwd_candidates:
            raise ValueError("No forward_* column found. Run eng.add_next_return() first.")
        forward_col = fwd_candidates[0]

    result = _std_factors(df, factor_cols, by)
    std_cols = [f"_std_{fc}" for fc in factor_cols]
    dates = sorted(df["date"].unique())

    # Precompute ICs for each factor-date
    ic_df = pd.DataFrame({"date": dates})
    for fc in factor_cols:
        ic_series = []
        for d in dates:
            sub = df[df["date"] == d]
            valid = sub[[fc, forward_col]].dropna()
            if len(valid) < 5:
                ic_series.append(np.nan)
            else:
                ic_series.append(valid[fc].corr(valid[forward_col]))
        ic_df[fc] = ic_series
    ic_df = ic_df.set_index("date")

    # Rolling IC weights per date
    composite = np.full(len(df), np.nan)
    dates_list = list(dates)
    for di, d in enumerate(dates_list):
        mask = df["date"] == d
        trailing = ic_df.iloc[max(0, di - window + 1):di + 1]
        raw_weights = trailing.mean().values  # mean IC per factor
        abs_sum = np.sum(np.abs(raw_weights))
        if abs_sum < 1e-10:
            continue
        weights = raw_weights / abs_sum
        for fi, fc in enumerate(factor_cols):
            composite[mask.values] = (
                np.nan_to_num(composite[mask.values], nan=0.0)
                + result.loc[mask, f"_std_{fc}"].values * weights[fi]
            )

    result[new_col] = composite
    result.drop(columns=std_cols, inplace=True)
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════════
# icir_weighted_composite
# ══════════════════════════════════════════════════════════════════════════════


def icir_weighted_composite(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    forward_col: Optional[str] = None,
    window: int = 60,
    by: str = "date",
    new_col: str = "composite_icir",
    append: bool = True,
) -> pd.DataFrame:
    """Weighted composite using trailing Information Ratio as weights.

    IR = mean(trailing IC) / std(trailing IC). Weights are set to 0
    when there are fewer than 12 observations in the trailing window.
    """
    validate_panel(df, extra_cols=factor_cols)
    if forward_col is None:
        fwd_candidates = [c for c in df.columns if c.startswith("forward_")]
        if not fwd_candidates:
            raise ValueError("No forward_* column found. Run eng.add_next_return() first.")
        forward_col = fwd_candidates[0]

    result = _std_factors(df, factor_cols, by)
    std_cols = [f"_std_{fc}" for fc in factor_cols]
    dates = sorted(df["date"].unique())

    # Precompute ICs
    ic_df = pd.DataFrame({"date": dates})
    for fc in factor_cols:
        ic_series = []
        for d in dates:
            sub = df[df["date"] == d]
            valid = sub[[fc, forward_col]].dropna()
            if len(valid) < 5:
                ic_series.append(np.nan)
            else:
                ic_series.append(valid[fc].corr(valid[forward_col]))
        ic_df[fc] = ic_series
    ic_df = ic_df.set_index("date")

    # Rolling ICIR weights per date
    composite = np.full(len(df), np.nan)
    dates_list = list(dates)
    for di, d in enumerate(dates_list):
        mask = df["date"] == d
        start = max(0, di - window + 1)
        trailing = ic_df.iloc[start:di + 1]
        raw_weights = []
        for fc in factor_cols:
            vals = trailing[fc].dropna()
            if len(vals) < 12:
                raw_weights.append(0.0)
            else:
                raw_weights.append(vals.mean() / max(vals.std(), 1e-10))
        raw_weights = np.array(raw_weights)
        abs_sum = np.sum(np.abs(raw_weights))
        if abs_sum < 1e-10:
            continue
        weights = raw_weights / abs_sum
        for fi, fc in enumerate(factor_cols):
            composite[mask.values] = (
                np.nan_to_num(composite[mask.values], nan=0.0)
                + result.loc[mask, f"_std_{fc}"].values * weights[fi]
            )

    result[new_col] = composite
    result.drop(columns=std_cols, inplace=True)
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════════
# pca_composite
# ══════════════════════════════════════════════════════════════════════════════


def pca_composite(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    n_components: int = 1,
    window: int = 252,
    min_periods: int = 60,
    by: str = "date",
    new_col: str = "composite_pca",
    append: bool = True,
) -> pd.DataFrame:
    """Composite using the first principal component of the factor matrix.

    Uses expanding window: all data up to and including the current date is
    used to fit the PCA, avoiding look-ahead bias.

    Parameters
    ----------
    n_components : int
        Number of PCs to project onto. Default 1.
    window : int
        Expanding window size. Currently unused (uses full expanding window).
    min_periods : int
        Minimum number of valid assets per cross-section.
    """
    validate_panel(df, extra_cols=factor_cols)
    result = _std_factors(df, factor_cols, by)
    std_cols = [f"_std_{fc}" for fc in factor_cols]
    dates = sorted(df["date"].unique())
    composite = np.full(len(df), np.nan)

    for d in dates:
        mask = df["date"] == d
        sub = result.loc[mask, std_cols].dropna()
        if len(sub) < min_periods or len(sub) < len(factor_cols):
            continue
        X = sub.values
        cov = np.cov(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        n_comp = min(n_components, len(factor_cols))
        proj = X @ eigvecs[:, :n_comp]
        comp_vals = proj.sum(axis=1) if n_comp > 1 else proj[:, 0]
        composite[sub.index] = comp_vals

    result[new_col] = composite
    result.drop(columns=std_cols, inplace=True)
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════════
# max_decay_composite
# ══════════════════════════════════════════════════════════════════════════════


def max_decay_composite(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    forward_col: Optional[str] = None,
    window: int = 60,
    decay: float = 0.5,
    by: str = "date",
    new_col: str = "composite_md",
    append: bool = True,
) -> pd.DataFrame:
    """Composite using only factors with positive trailing IC, time-decayed.

    Only factors whose trailing mean IC > 0 are used. The IC is weighted
    by a linearly decaying scheme within the trailing window.

    Parameters
    ----------
    decay : float
        Decay rate. Larger values give more weight to recent observations.
    """
    validate_panel(df, extra_cols=factor_cols)
    if forward_col is None:
        fwd_candidates = [c for c in df.columns if c.startswith("forward_")]
        if not fwd_candidates:
            raise ValueError("No forward_* column found. Run eng.add_next_return() first.")
        forward_col = fwd_candidates[0]

    result = _std_factors(df, factor_cols, by)
    std_cols = [f"_std_{fc}" for fc in factor_cols]
    dates = sorted(df["date"].unique())

    # Precompute ICs
    ic_df = pd.DataFrame({"date": dates})
    for fc in factor_cols:
        ic_series = []
        for d in dates:
            sub = df[df["date"] == d]
            valid = sub[[fc, forward_col]].dropna()
            if len(valid) < 5:
                ic_series.append(np.nan)
            else:
                ic_series.append(valid[fc].corr(valid[forward_col]))
        ic_df[fc] = ic_series
    ic_df = ic_df.set_index("date")

    composite = np.full(len(df), np.nan)
    dates_list = list(dates)
    for di, d in enumerate(dates_list):
        mask = df["date"] == d
        start = max(0, di - window + 1)
        trailing = ic_df.iloc[start:di + 1]
        n_trail = len(trailing)
        if n_trail < 5:
            continue
        # Linearly decaying weights: newest gets weight 1, oldest gets decay
        decay_weights = np.linspace(decay, 1.0, n_trail)
        raw_weights = []
        for fc in factor_cols:
            ic_vals = trailing[fc].values
            valid_mask = ~np.isnan(ic_vals)
            if valid_mask.sum() < 5:
                raw_weights.append(0.0)
            else:
                w_mean = np.average(ic_vals[valid_mask], weights=decay_weights[valid_mask])
                raw_weights.append(max(w_mean, 0.0))
        raw_weights = np.array(raw_weights)
        abs_sum = np.sum(np.abs(raw_weights))
        if abs_sum < 1e-10:
            continue
        weights = raw_weights / abs_sum
        for fi, fc in enumerate(factor_cols):
            composite[mask.values] = (
                np.nan_to_num(composite[mask.values], nan=0.0)
                + result.loc[mask, f"_std_{fc}"].values * weights[fi]
            )

    result[new_col] = composite
    result.drop(columns=std_cols, inplace=True)
    return slim_output(result, new_col, append)
