"""Visualization functions — eBacktestCraft plot_* equivalent.

Each function takes a backtest result or equity curve DataFrame
and returns a matplotlib Figure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ── Global style ─────────────────────────────────────────────────────────
QUANT_STYLE = {
    "figure.facecolor": "#f8f9fa",
    "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#dee2e6",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#adb5bd",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
}


def theme_quant():
    """Apply the eBacktestCraft visual theme globally."""
    for k, v in QUANT_STYLE.items():
        plt.rcParams[k] = v


# ══════════════════════════════════════════════════════════════════════════
# Equity Curve
# ══════════════════════════════════════════════════════════════════════════


def plot_df(
    df: pd.DataFrame,
    benchmark_curve: Optional[pd.DataFrame] = None,
    benchmark_label: str = "Benchmark",
    title: str = "Equity Curve",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot cumulative equity curve.

    Parameters
    ----------
    df : DataFrame
        Must have ``date`` and ``nav`` columns.
    benchmark_curve : DataFrame, optional
        Benchmark equity curve for comparison.
    """
    theme_quant()
    fig, ax = plt.subplots(figsize=figsize)

    eq = df.copy()
    if "date" in eq.columns:
        eq = eq.set_index("date")

    # Cumulative return as percentage
    init_nav = eq["nav"].iloc[0] if "nav" in eq.columns else 100_000
    cum_ret = (eq["nav"] / init_nav - 1) * 100 if "nav" in eq.columns else eq["cumulative_return"] * 100

    ax.plot(eq.index, cum_ret.values, linewidth=1.5, color="#2196F3", label="Strategy", zorder=3)
    ax.fill_between(eq.index, 0, cum_ret.values, alpha=0.1, color="#2196F3")

    if benchmark_curve is not None:
        bm = benchmark_curve.copy()
        if "date" in bm.columns:
            bm = bm.set_index("date")
        bm_nav = bm["nav"].iloc[0] if "nav" in bm.columns else 100_000
        bm_ret = (bm["nav"] / bm_nav - 1) * 100
        ax.plot(bm.index, bm_ret.values, linewidth=1.2, color="#9E9E9E",
                linestyle="--", label=benchmark_label, zorder=2)

    ax.axhline(y=0, color="#dee2e6", linewidth=0.8, zorder=1)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f%%"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Drawdown
# ══════════════════════════════════════════════════════════════════════════


def plot_drawdown(
    df: pd.DataFrame,
    title: str = "Drawdown",
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot drawdown chart.

    Parameters
    ----------
    df : DataFrame
        Must have ``date`` and ``nav`` columns, or ``drawdown`` column.
    """
    theme_quant()
    fig, ax = plt.subplots(figsize=figsize)

    eq = df.copy()
    if "date" in eq.columns:
        eq = eq.set_index("date")

    if "drawdown" in eq.columns:
        dd = eq["drawdown"].values * 100
    else:
        nav = eq["nav"].values
        cummax = np.maximum.accumulate(nav)
        dd = (nav - cummax) / cummax * 100

    ax.fill_between(eq.index, 0, dd, alpha=0.35, color="#E53935", zorder=2)
    ax.plot(eq.index, dd, linewidth=0.8, color="#C62828", zorder=3)
    ax.axhline(y=0, color="#dee2e6", linewidth=0.8, zorder=1)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.invert_yaxis()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Combined Return + Drawdown
# ══════════════════════════════════════════════════════════════════════════


def plot_return_drawdown(
    df: pd.DataFrame,
    title: str = "Strategy Performance",
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Combined equity curve (top) and drawdown (bottom) chart."""
    theme_quant()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1.5],
                                    sharex=True)

    eq = df.copy()
    if "date" in eq.columns:
        eq = eq.set_index("date")

    # Top: equity curve
    nav = eq["nav"].values
    init_nav = nav[0]
    cum_ret = (nav / init_nav - 1) * 100
    ax1.plot(eq.index, cum_ret, linewidth=1.5, color="#2196F3", zorder=3)
    ax1.fill_between(eq.index, 0, cum_ret, alpha=0.1, color="#2196F3")
    ax1.axhline(y=0, color="#dee2e6", linewidth=0.8, zorder=1)
    ax1.set_title(title, fontweight="bold")
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f%%"))

    # Bottom: drawdown
    cummax = np.maximum.accumulate(nav)
    dd = (nav - cummax) / cummax * 100
    ax2.fill_between(eq.index, 0, dd, alpha=0.35, color="#E53935", zorder=2)
    ax2.plot(eq.index, dd, linewidth=0.8, color="#C62828", zorder=3)
    ax2.axhline(y=0, color="#dee2e6", linewidth=0.8, zorder=1)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.invert_yaxis()
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Daily Returns Distribution
# ══════════════════════════════════════════════════════════════════════════


def plot_return_dist(
    df: pd.DataFrame,
    bins: int = 50,
    title: str = "Daily Return Distribution",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram + KDE of daily returns."""
    theme_quant()
    fig, ax = plt.subplots(figsize=figsize)

    eq = df.copy()
    if "return" in eq.columns:
        rets = eq["return"].dropna() * 100  # Convert to percentage
    else:
        nav = eq["nav"].values
        rets = (np.diff(nav) / nav[:-1]) * 100

    ax.hist(rets, bins=bins, density=True, alpha=0.55, color="#2196F3",
            edgecolor="white", linewidth=0.5)
    sns.kdeplot(x=rets, ax=ax, color="#1565C0", linewidth=2)

    # Stats annotations
    mu = np.mean(rets)
    sigma = np.std(rets, ddof=1)
    skew = pd.Series(rets).skew()
    kurt = pd.Series(rets).kurtosis()

    stats_text = (
        f"Mean: {mu:+.3f}%  |  Std: {sigma:.3f}%  |  "
        f"Skew: {skew:+.2f}  |  Kurt: {kurt:+.2f}"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax.axvline(x=0, color="#dee2e6", linewidth=1, linestyle="--")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Monthly Returns Heatmap
# ══════════════════════════════════════════════════════════════════════════


def plot_monthly_return(
    df: pd.DataFrame,
    title: str = "Monthly Returns (%)",
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Monthly returns heatmap table."""
    theme_quant()
    eq = df.copy()

    if "date" not in eq.columns:
        eq = eq.reset_index()
    eq["date"] = pd.to_datetime(eq["date"])

    # Resample to monthly
    eq["year"] = eq["date"].dt.year
    eq["month"] = eq["date"].dt.month

    if "return" in eq.columns:
        monthly = eq.groupby(["year", "month"])["return"].apply(
            lambda x: (1 + x).prod() - 1
        ).unstack() * 100
    else:
        nav = eq.set_index("date")["nav"]
        monthly_ret = nav.resample("ME").last().pct_change() * 100
        monthly_ret.index = pd.MultiIndex.from_arrays(
            [monthly_ret.index.year, monthly_ret.index.month]
        )
        monthly = monthly_ret.unstack()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        monthly, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
        linewidths=1, linecolor="white", cbar_kws={"label": "Return (%)"},
        ax=ax, annot_kws={"fontsize": 9},
    )

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_yticklabels([f"{int(y):.0f}" for y in monthly.index], rotation=0)
    ax.set_xticklabels([month_names[int(m) - 1] for m in monthly.columns])
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Benchmark Comparison
# ══════════════════════════════════════════════════════════════════════════


def plot_benchmark_compare(
    results: dict[str, pd.DataFrame],
    title: str = "Strategy vs Benchmarks",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Overlay multiple equity curves for comparison.

    Parameters
    ----------
    results : dict
        Mapping of strategy name → df DataFrame.
    """
    theme_quant()
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#607D8B"]

    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, eq) in enumerate(results.items()):
        df = eq.copy()
        if "date" in df.columns:
            df = df.set_index("date")
        nav = df["nav"].values
        cum_ret = (nav / nav[0] - 1) * 100
        color = colors[i % len(colors)]
        linestyle = "-" if i == 0 else "--"
        ax.plot(df.index, cum_ret, linewidth=1.5, color=color,
                linestyle=linestyle, label=name, zorder=3)

    ax.axhline(y=0, color="#dee2e6", linewidth=0.8, zorder=1)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f%%"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_all(
    df: pd.DataFrame,
    benchmark_curve: Optional[pd.DataFrame] = None,
    title_prefix: str = "",
    save_dir: Optional[str] = None,
) -> dict[str, plt.Figure]:
    """Generate all standard backtest visualization plots.

    Returns a dict of ``{name: Figure}``.
    """
    figs = {}
    figs["equity"] = plot_df(
        df, benchmark_curve,
        title=f"{title_prefix} Equity Curve" if title_prefix else "Equity Curve",
        save_path=f"{save_dir}/df.png" if save_dir else None,
    )
    figs["drawdown"] = plot_drawdown(
        df,
        title=f"{title_prefix} Drawdown" if title_prefix else "Drawdown",
        save_path=f"{save_dir}/drawdown.png" if save_dir else None,
    )
    figs["combined"] = plot_return_drawdown(
        df,
        title=f"{title_prefix} Performance" if title_prefix else "Strategy Performance",
        save_path=f"{save_dir}/performance.png" if save_dir else None,
    )
    figs["dist"] = plot_return_dist(
        df,
        save_path=f"{save_dir}/return_dist.png" if save_dir else None,
    )
    figs["monthly"] = plot_monthly_return(
        df,
        save_path=f"{save_dir}/monthly_returns.png" if save_dir else None,
    )
    return figs


# Public alias — ebacktestcraft/__init__.py and external callers (run_viz.py)
# use this name; plot_df is the internal name used within this module.
plot_equity_curve = plot_df
