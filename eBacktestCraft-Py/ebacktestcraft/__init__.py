"""eBacktestCraft — Professional backtesting framework for long-format panel DataFrames.

Signal generation, weight schemes, event-driven engine,
stop-loss/take-profit, performance analytics, and strategy enhancement.

Usage::

    import ebacktestcraft as ebc

    cfg = ebc.Config(init_capital=100_000, rebalance_cycle="monthly")
    df = ebc.signal(df, factor_col="mom_20", method="quantile")
    df = ebc.equal_weight(df)
    result = ebc.run(df, config=cfg)
    metrics = ebc.performance_analysis(result.equity_curve)
"""

from ebacktestcraft.analytics import performance_analysis
from ebacktestcraft.benchmark import (
    buy_and_hold_benchmark,
    compare_benchmarks,
    equal_weight_benchmark,
    index_benchmark,
)
from ebacktestcraft.config import BacktestConfig
from ebacktestcraft.engine import BacktestResult, run
from ebacktestcraft.param_scan import (
    best_params,
    param_grid,
    rank_param_scan,
    run_param_scan,
    sensitivity_table,
)
from ebacktestcraft.signals import signal
from ebacktestcraft.weights import equal_weight, fixed_weight, norm_weight

# ── Plotting (optional, requires matplotlib/seaborn) ──
try:
    from ebacktestcraft.plot import (
        plot_all,
        plot_benchmark_compare,
        plot_drawdown,
        plot_equity_curve,
        plot_monthly_return,
        plot_return_dist,
        plot_return_drawdown,
        theme_quant,
    )
    _has_plot = True
except ImportError:
    _has_plot = False

# ── Strategy enhancement ──
from ebacktestcraft import enhance
from ebacktestcraft.enhance.weights import (
    confidence_weight,
    erp_weight,
    target_vol_weight,
    vol_parity_weight,
)
from ebacktestcraft.enhance.signals import (
    persistent_signal,
    quantile_signal,
    smoothed_signal,
)
from ebacktestcraft.enhance.risk import (
    apply_vol_target,
    compute_turnover,
)

# Alias for user convenience
Config = BacktestConfig

__all__ = [
    "Config",
    "BacktestConfig",
    "BacktestResult",
    "run",
    "signal",
    "equal_weight",
    "fixed_weight",
    "norm_weight",
    "performance_analysis",
    "equal_weight_benchmark",
    "buy_and_hold_benchmark",
    "index_benchmark",
    "compare_benchmarks",
    "plot_equity_curve",
    "plot_drawdown",
    "plot_return_drawdown",
    "plot_return_dist",
    "plot_monthly_return",
    "plot_benchmark_compare",
    "plot_all",
    "theme_quant",
    # Enhancement
    "enhance",
    "vol_parity_weight",
    "target_vol_weight",
    "erp_weight",
    "confidence_weight",
    "quantile_signal",
    "persistent_signal",
    "smoothed_signal",
    "apply_vol_target",
    "compute_turnover",
]
__version__ = "0.1.0"
