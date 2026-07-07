"""Performance analytics — eBacktestCraft Performance_Analyze equivalent."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def performance_analysis(
    df: pd.DataFrame,
    transactions: Optional[pd.DataFrame] = None,
    init_capital: float = 100_000.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Compute comprehensive performance metrics.

    Parameters
    ----------
    df : DataFrame
        Must have ``date``, ``nav``, ``return`` columns.
    transactions : DataFrame, optional
        Trades for turnover/win-rate analysis.
    init_capital : float
        Starting capital.
    risk_free_rate : float
        Annual risk-free rate.
    periods_per_year : int
        252 for daily, 12 for monthly.

    Returns
    -------
    dict
        Performance metrics.
    """
    eq = df.copy()
    if "nav" not in eq.columns:
        raise ValueError("df must have 'nav' column")

    nav = eq["nav"].values
    n = len(eq)

    # Basic
    final_nav = nav[-1] if n > 0 else init_capital
    total_return = (final_nav - init_capital) / init_capital
    annual_return = (final_nav / init_capital) ** (periods_per_year / n) - 1 if n > 0 else 0.0

    # Daily returns
    if "return" in eq.columns:
        daily_rets = eq["return"].dropna().values
    else:
        daily_rets = np.diff(nav) / nav[:-1]

    # Volatility
    daily_vol = np.std(daily_rets, ddof=1) if len(daily_rets) > 1 else 0.0
    annual_vol = daily_vol * np.sqrt(periods_per_year)

    # Sharpe
    excess = daily_rets - risk_free_rate / periods_per_year
    sharpe = np.sqrt(periods_per_year) * excess.mean() / daily_vol if daily_vol > 0 else 0.0

    # Sortino
    downside = daily_rets[daily_rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
    sortino = np.sqrt(periods_per_year) * excess.mean() / downside_std if downside_std > 0 else 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(nav)
    drawdowns = (nav - cummax) / cummax
    max_dd = drawdowns.min()
    max_dd_idx = np.argmin(drawdowns)
    max_dd_date = eq["date"].iloc[max_dd_idx] if "date" in eq.columns and max_dd_idx < len(eq) else None

    # Calmar
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 0 else 0.0

    # Win rate
    win_rate = (daily_rets > 0).mean() if len(daily_rets) > 0 else 0.0

    # Best / worst day
    best_day = daily_rets.max() if len(daily_rets) > 0 else 0.0
    worst_day = daily_rets.min() if len(daily_rets) > 0 else 0.0

    # Trade stats
    n_trades = len(transactions) if transactions is not None else 0

    # Cumulative return for plotting
    eq["cumulative_return"] = nav / init_capital - 1.0
    eq["drawdown"] = drawdowns

    return {
        "total_return_pct": total_return * 100,
        "annual_return_pct": annual_return * 100,
        "annual_volatility_pct": annual_vol * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown_pct": max_dd * 100,
        "max_drawdown_date": str(max_dd_date),
        "win_rate_pct": win_rate * 100,
        "best_day_pct": best_day * 100,
        "worst_day_pct": worst_day * 100,
        "n_trades": n_trades,
        "n_days": n,
        "init_capital": init_capital,
        "final_nav": final_nav,
        "df": eq,
    }
