"""eFactorCraft — Factor engineering pipeline for long-format panel DataFrames.

Preprocessing, neutralization, ranking, IC/IR analysis, factor synthesis,
factor selection, factor timing, and data providers.

Usage::

    from efactorcraft import get_data
    import efactorcraft as efc

    df = get_data(universe, "2023-01-01", "2024-01-01")
    df = efc.winsorize(df, factor_col="mom_20")
    df = efc.standardize(df, factor_col="mom_20")
    df = efc.industry_neutralize(df, factor_col="mom_20", industry_col="industry")
    ic = efc.ic_analysis(df, factor_cols=["mom_20"])
"""

# ── Engineering: preprocessing ──
from efactorcraft.preprocess import (
    factor_preprocess,
    industry_neutralize,
    size_neutralize,
    standardize,
    winsorize,
)

# ── Engineering: analysis ──
from efactorcraft.analysis import (
    add_next_return,
    ic_analysis,
    ir_analysis,
    quantile_analysis,
)

# ── Engineering: rank ──
from efactorcraft.rank import (
    consecutive_days,
    quantile_flag,
    quantile_rank,
)

# ── Engineering: data ──
from efactorcraft.data import get_data

# ── Synthesis ──
from efactorcraft.synthesis import (
    equal_weighted_composite,
    ic_weighted_composite,
    icir_weighted_composite,
    max_decay_composite,
    pca_composite,
    rank_weighted_composite,
)

# ── Selection ──
from efactorcraft.selection import (
    correlation_screen,
    factor_report,
    ic_screen,
    select_top,
    stability_screen,
)

# ── Timing ──
from efactorcraft.timing import (
    adaptive_composite,
    regime_detect,
    timing_weight,
    trend_filter,
    vol_filter,
)

# ── Providers ──
from efactorcraft import providers

__all__ = [
    # Preprocess
    "winsorize",
    "standardize",
    "industry_neutralize",
    "size_neutralize",
    "factor_preprocess",
    # Analysis
    "add_next_return",
    "ic_analysis",
    "ir_analysis",
    "quantile_analysis",
    # Rank
    "quantile_rank",
    "quantile_flag",
    "consecutive_days",
    # Data
    "get_data",
    # Synthesis
    "equal_weighted_composite",
    "ic_weighted_composite",
    "icir_weighted_composite",
    "max_decay_composite",
    "pca_composite",
    "rank_weighted_composite",
    # Selection
    "correlation_screen",
    "factor_report",
    "ic_screen",
    "select_top",
    "stability_screen",
    # Timing
    "adaptive_composite",
    "regime_detect",
    "timing_weight",
    "trend_filter",
    "vol_filter",
    # Providers
    "providers",
]
__version__ = "0.1.0"
