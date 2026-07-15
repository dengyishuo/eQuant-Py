"""Backtest configuration — eBacktestCraft equivalent.

Replaces the R list+functional-setter pattern with a Python dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class BacktestConfig:
    """All parameters for the backtesting engine.

    Usage::

        cfg = BacktestConfig(init_capital=100_000, rebalance_cycle="monthly")
        cfg.set(lot_size=200)  # chainable
    """

    # ── Core ──
    weight_col: str = "weight"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    exec_price_col: str = "open"
    eval_price_col: str = "adjusted"
    init_capital: float = 100_000.0
    lot_size: int = 100
    fee_rate: float = 0.0003
    stamp_tax: float = 0.001
    slippage_rate: float = 0.001
    min_weight: float = 1e-6
    single_max_weight: float = 0.95
    global_max_hold_pct: float = 1.0

    # ── Component Stop Loss ──
    enable_component_stop_loss: bool = False
    component_stop_loss_type: Literal["fixed", "trailing_fixed", "trailing_atr", "trailing_vol", "trailing_log", "stop_limit"] = "fixed"
    fixed_component_sl_ratio: float = 0.1
    trailing_fixed_component_sl_ratio: float = 0.1
    atr_n_component: int = 14
    atr_k_component: float = 2.0
    vol_n_component: int = 20
    vol_sigma_component: float = 2.0
    log_vol_n_component: int = 20
    log_vol_sigma_component: float = 2.0

    # ── Portfolio Stop Loss ──
    enable_portfolio_stop_loss: bool = False
    portfolio_stop_loss_type: Literal["fixed", "trailing_fixed", "trailing_atr", "trailing_vol", "trailing_log", "stop_limit"] = "fixed"
    fixed_portfolio_sl_ratio: float = 0.1
    trailing_fixed_portfolio_sl_ratio: float = 0.1
    atr_n_portfolio: int = 14
    atr_k_portfolio: float = 2.0
    vol_n_portfolio: int = 20
    vol_sigma_portfolio: float = 2.0
    log_vol_n_portfolio: int = 20
    log_vol_sigma_portfolio: float = 2.0

    # ── Component Take Profit ──
    enable_component_take_profit: bool = False
    component_take_profit_type: Literal["fixed", "trailing_fixed", "trailing_atr", "trailing_vol", "trailing_log", "stop_limit"] = "fixed"
    fixed_component_tp_ratio: float = 0.1
    trailing_fixed_component_tp_ratio: float = 0.1
    atr_k_component_tp: float = 2.0
    vol_sigma_component_tp: float = 2.0
    log_vol_sigma_component_tp: float = 2.0

    # ── Portfolio Take Profit ──
    enable_portfolio_take_profit: bool = False
    portfolio_take_profit_type: Literal["fixed", "trailing_fixed", "trailing_atr", "trailing_vol", "trailing_log", "stop_limit"] = "fixed"
    fixed_portfolio_tp_ratio: float = 0.1
    trailing_fixed_portfolio_tp_ratio: float = 0.1
    atr_k_portfolio_tp: float = 2.0
    vol_sigma_portfolio_tp: float = 2.0
    log_vol_sigma_portfolio_tp: float = 2.0

    # ── Stop-Limit execution gap ─────────────────────────────────────────────
    # For stop_limit SL: limit floor = trigger * (1 - gap).  Won't fill if
    # open gaps below this floor (price skipped the limit level entirely).
    # For stop_limit TP: limit ceiling = trigger * (1 + gap).  Won't fill if
    # open gaps above this ceiling (already past the bracket window).
    stop_limit_component_sl_gap: float = 0.005
    stop_limit_component_tp_gap: float = 0.005
    stop_limit_portfolio_sl_gap: float = 0.005
    stop_limit_portfolio_tp_gap: float = 0.005

    # ── OCO (One-Cancels-Other) bracket ──────────────────────────────────────
    # When True, stop-loss and take-profit are treated as a linked pair:
    # whichever fires first closes the position; the other is cancelled.
    # Equivalent to enabling both SL and TP simultaneously.
    enable_oco_component: bool = False
    enable_oco_portfolio: bool = False

    # ── Rebalancing ──
    rebalance_mode: Literal["calendar", "weight_shift", "hybrid"] = "calendar"
    rebalance_cycle: str = "quarterly"
    weight_change_threshold: float = 0.01
    skip_suspended: bool = True

    def set(self, **kwargs) -> "BacktestConfig":
        """Set multiple parameters; returns self for chaining."""
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown config parameter: {k}")
            setattr(self, k, v)
        return self

    def to_dict(self) -> dict:
        """Return all parameters as a plain dict."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
