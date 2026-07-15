"""Miscellaneous indicators — remaining eTTR functions.

Adds indicators not covered in trend/momentum/volatility/volume/patterns:
- Growth rate, adjusted ratios, rolling SFM (slope of linear fit)
- Aroon (panel wrapper), TD Setup/Countdown (DeMark indicators)
- Lag utilities, NA checking, index alignment, performance calculation
"""

from __future__ import annotations

from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd

from ettr._panel import _resolve_col
from ettr._rolling import aroon_max, roll_sd
from equant.utils.panel import slim_output, validate_panel


# ══════════════════════════════════════════════════════════════════════════
# Growth Rate
# ══════════════════════════════════════════════════════════════════════════


def growth(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: Union[int, Sequence[int]] = 1,
    type: str = "continuous",
    new_col: str = "growth",
    append: bool = True,
) -> pd.DataFrame:
    """Period-over-period growth rate.

    ``growth = (x[t] - x[t-n]) / |x[t-n]|``
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    ns = [n] if isinstance(n, int) else list(n)
    out_cols = [f"{new_col}_{p}" for p in ns]

    result = df.copy()
    for period in ns:
        cname = f"{new_col}_{period}"
        result[cname] = np.nan
        for _code, idx in result.groupby("code", sort=False).groups.items():
            sub = result.loc[idx].sort_values("date")
            vals = sub[col].values.astype(np.float64)
            if type == "continuous":
                shifted = np.roll(vals, period)
                shifted[:period] = np.nan
                result.loc[sub.index, cname] = (vals - shifted) / np.maximum(np.abs(shifted), 1e-15)
            else:
                result.loc[sub.index, cname] = pd.Series(vals).pct_change(period).values

    return slim_output(result, out_cols, append)


# ══════════════════════════════════════════════════════════════════════════
# Adjusted Ratios (split/dividend adjustment factors)
# ══════════════════════════════════════════════════════════════════════════


def adj_ratios(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    adjusted_col: Optional[str] = None,
    new_col: str = "adj_ratio",
    append: bool = True,
) -> pd.DataFrame:
    """Compute split/dividend adjustment ratio.

    ``adj_ratio = adjusted / close``
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    adj = _resolve_col(df, "adjusted", adjusted_col)
    result = df.copy()
    result[new_col] = np.where(
        result[col] > 0,
        result[adj] / result[col],
        1.0,
    )
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════
# Rolling SFM — Single Factor Model slope over rolling window
# ══════════════════════════════════════════════════════════════════════════


def roll_sfm(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    n: int = 60,
    new_col: str = "sfm",
    append: bool = True,
) -> pd.DataFrame:
    """Rolling single-factor model slope.

    Regresses price against ``1, 2, ..., n`` (time index) within each
    rolling window. The slope measures the linear trend strength.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    out_cols = [f"{new_col}_alpha", f"{new_col}_beta", f"{new_col}_r2"]
    x = np.arange(1, n + 1, dtype=np.float64)
    x_mean = x.mean()
    sxx = ((x - x_mean) ** 2).sum()

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        vals = sub[col].values.astype(np.float64)

        for i in range(n - 1, len(vals)):
            y = vals[i - n + 1 : i + 1]
            valid = ~np.isnan(y)
            if valid.sum() < n // 2:
                continue
            yv = y[valid]
            xv = x[valid]
            ym = yv.mean()
            sxy = ((xv - x_mean) * (yv - ym)).sum()
            beta = sxy / sxx
            alpha = ym - beta * x_mean
            y_pred = alpha + beta * xv
            ss_res = ((yv - y_pred) ** 2).sum()
            ss_tot = ((yv - ym) ** 2).sum()
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            result.loc[sub.index[i], out_cols[0]] = alpha
            result.loc[sub.index[i], out_cols[1]] = beta
            result.loc[sub.index[i], out_cols[2]] = r2

    return slim_output(result, out_cols, append)


# ══════════════════════════════════════════════════════════════════════════
# Aroon (panel wrapper around numba aroon_max)
# ══════════════════════════════════════════════════════════════════════════


def aroon(
    df: pd.DataFrame,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    n: int = 25,
    new_col: str = "Aroon",
    append: bool = True,
) -> pd.DataFrame:
    """Aroon indicator — measures time since highest high / lowest low.

    Produces ``{new_col}_up``, ``{new_col}_down``, ``{new_col}_osc``.
    """
    validate_panel(df)
    hcol = _resolve_col(df, "high", high_col)
    lcol = _resolve_col(df, "low", low_col)
    out_cols = [f"{new_col}_up", f"{new_col}_down", f"{new_col}_osc"]

    result = df.copy()
    for c in out_cols:
        result[c] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        h = sub[hcol].values.astype(np.float64)
        l = sub[lcol].values.astype(np.float64)
        up, down, osc = aroon_max(h, l, n)
        result.loc[sub.index, out_cols[0]] = up
        result.loc[sub.index, out_cols[1]] = down
        result.loc[sub.index, out_cols[2]] = osc

    return slim_output(result, out_cols, append)


# ══════════════════════════════════════════════════════════════════════════
# TD Sequential (DeMark) — Setup
# ══════════════════════════════════════════════════════════════════════════


def td_setup(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    new_col: str = "TD_Setup",
    append: bool = True,
) -> pd.DataFrame:
    """Tom DeMark Setup — counts consecutive bars comparing to 4 bars ago.

    Buy setup: consecutive closes < close[4 bars ago], up to 9.
    Sell setup: consecutive closes > close[4 bars ago], up to 9.

    Positive = buy setup count, Negative = sell setup count.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    result = df.copy()
    result[new_col] = 0

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        c = sub[col].values.astype(np.float64)
        td = np.zeros(len(c), dtype=int)
        count = 0
        direction = 0  # 0=neutral, 1=buy, -1=sell

        for i in range(4, len(c)):
            if np.isnan(c[i]) or np.isnan(c[i - 4]):
                count = 0
                direction = 0
                continue

            if c[i] < c[i - 4]:  # Buy setup condition
                if direction == 1:
                    count += 1
                else:
                    count = 1
                    direction = 1
            elif c[i] > c[i - 4]:  # Sell setup condition
                if direction == -1:
                    count -= 1
                else:
                    count = -1
                    direction = -1
            else:
                count = 0
                direction = 0

            td[i] = count

        result.loc[sub.index, new_col] = td

    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════
# TD Countdown
# ══════════════════════════════════════════════════════════════════════════


def td_countdown(
    df: pd.DataFrame,
    close_col: Optional[str] = None,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    new_col: str = "TD_Countdown",
    append: bool = True,
) -> pd.DataFrame:
    """Tom DeMark Countdown — 13-bar exhaustion count after Setup completion.

    Counts 13 bars where conditions are met after a TD Setup of 9.
    Positive = buy countdown, Negative = sell countdown.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    hcol = _resolve_col(df, "high", high_col) if high_col else "high"
    lcol = _resolve_col(df, "low", low_col) if low_col else "low"
    result = df.copy()
    result[new_col] = 0

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        c = sub[col].values.astype(np.float64)
        h = sub[hcol].values if hcol in sub.columns else c
        l = sub[lcol].values if lcol in sub.columns else c
        td = np.zeros(len(c), dtype=int)

        # First pass: find setups
        setup = np.zeros(len(c), dtype=int)
        dir_s = 0
        cnt = 0
        for i in range(4, len(c)):
            if np.isnan(c[i]) or np.isnan(c[i - 4]):
                cnt = 0; dir_s = 0; continue
            if c[i] < c[i - 4]:
                cnt = cnt + 1 if dir_s == 1 else 1; dir_s = 1
            elif c[i] > c[i - 4]:
                cnt = cnt + 1 if dir_s == -1 else -1; dir_s = -1
            else:
                cnt = 0; dir_s = 0
            if abs(cnt) >= 9:
                setup[i] = 1 if cnt > 0 else -1

        # Second pass: find countdown after setup
        for i in range(len(c)):
            if setup[i] != 0:
                direction = setup[i]
                cd_cnt = 0
                for j in range(i + 1, len(c)):
                    if np.isnan(c[j]):
                        continue
                    if direction == 1:  # Buy countdown
                        if c[j] >= h[j - 2] if j >= 3 else True:
                            cd_cnt += 1
                    else:  # Sell countdown
                        if j >= 3 and c[j] <= l[j - 2]:
                            cd_cnt += 1
                    td[j] = cd_cnt * direction
                    if cd_cnt >= 13:
                        break

        result.loc[sub.index, new_col] = td

    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════
# NA Check — detect missing values in a series
# ══════════════════════════════════════════════════════════════════════════


def na_check(
    df: pd.DataFrame,
    col: Optional[str] = None,
    new_col: str = "is_na",
    append: bool = True,
) -> pd.DataFrame:
    """Flag rows where the target column is NaN."""
    validate_panel(df)
    if col is None:
        col = df.select_dtypes(include=[np.number]).columns[0]
    result = df.copy()
    result[new_col] = result[col].isna().astype(int)
    return slim_output(result, new_col, append)


# ══════════════════════════════════════════════════════════════════════════
# Lags — add lagged columns
# ══════════════════════════════════════════════════════════════════════════


def lags(
    df: pd.DataFrame,
    col: str,
    n: Union[int, Sequence[int]] = 1,
    new_col: str = "lag",
    append: bool = True,
) -> pd.DataFrame:
    """Add lagged columns of a variable.

    Parameters
    ----------
    n : int or sequence
        Lag period(s). 1 = one period back.
    """
    validate_panel(df)
    ns = [n] if isinstance(n, int) else list(n)
    out_cols = [f"{new_col}_{p}" for p in ns]

    result = df.copy()
    for period in ns:
        cname = f"{new_col}_{period}"
        result[cname] = np.nan
        for _code, idx in result.groupby("code", sort=False).groups.items():
            sub = result.loc[idx].sort_values("date")
            vals = sub[col].values
            shifted = np.roll(vals, period)
            shifted[:period] = np.nan
            result.loc[sub.index, cname] = shifted

    return slim_output(result, out_cols, append)


# ══════════════════════════════════════════════════════════════════════════
# Align with Index — align an asset to an index's trading days
# ══════════════════════════════════════════════════════════════════════════


def align_with_index(
    df: pd.DataFrame,
    index_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Filter *df* to only include dates present in *index_df*."""
    if date_col not in df.columns or date_col not in index_df.columns:
        raise ValueError(f"Both DataFrames must have '{date_col}' column")
    valid_dates = set(index_df[date_col].unique())
    return df[df[date_col].isin(valid_dates)].copy()


# ══════════════════════════════════════════════════════════════════════════
# Calculate Performance — summary statistics for a price series
# ══════════════════════════════════════════════════════════════════════════


def calculate_performance(
    df: pd.DataFrame,
    close_col: str = "close",
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Calculate performance metrics for a price series.

    Returns a dict of summary statistics.
    """
    validate_panel(df)
    vals = df[close_col].values.astype(np.float64)
    rets = np.full(len(vals), np.nan)
    rets[1:] = vals[1:] / np.maximum(np.abs(vals[:-1]), 1e-15) - 1.0
    rets = rets[~np.isnan(rets)]

    if len(rets) < 2:
        return {"n_obs": len(rets)}

    total_ret = (vals[-1] / vals[0] - 1) * 100
    ann_ret = ((1 + total_ret / 100) ** (periods_per_year / len(rets)) - 1) * 100
    ann_vol = np.std(rets, ddof=1) * np.sqrt(periods_per_year) * 100
    sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else 0.0

    cummax = np.maximum.accumulate(vals)
    drawdowns = (vals - cummax) / cummax
    max_dd = drawdowns.min() * 100

    win_rate = (rets > 0).mean() * 100

    return {
        "n_obs": len(rets),
        "total_return_pct": total_ret,
        "annual_return_pct": ann_ret,
        "annual_vol_pct": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "win_rate_pct": win_rate,
        "best_day_pct": rets.max() * 100,
        "worst_day_pct": rets.min() * 100,
    }
