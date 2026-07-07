"""Event-driven backtesting engine — eBacktestCraft .run_backtest equivalent.

Implements day-by-day portfolio simulation with:
- Multi-asset position tracking
- Calendar / weight_shift / hybrid rebalancing
- Component and portfolio-level stop-loss / take-profit
- Transaction costs (fees, stamp tax, slippage)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ebacktestcraft.config import BacktestConfig


@dataclass
class BacktestResult:
    """Container for backtest output."""

    daily_positions: pd.DataFrame
    equity_curve: pd.DataFrame
    transactions: pd.DataFrame
    config: dict
    summary: dict = field(default_factory=dict)


def run(
    df: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    weight_col: str = "weight",
    **kwargs,
) -> BacktestResult:
    """Run a multi-asset backtest.

    Parameters
    ----------
    df : DataFrame
        Long-format panel with OHLCV data and a weight column.
    config : BacktestConfig, optional
        Configuration object. If None, uses defaults.
    weight_col : str
        Column name containing target weights.
    **kwargs
        Override individual config parameters.
    """
    if config is None:
        config = BacktestConfig()
    if kwargs:
        config = BacktestConfig(**{**config.to_dict(), **kwargs})

    return _run_engine(df, config, weight_col)


def _run_engine(df: pd.DataFrame, cfg: BacktestConfig, weight_col: str) -> BacktestResult:
    """Internal backtesting engine."""

    # ── 1. Column normalization ──────────────────────────────────────────
    col_map = {
        "date": ["Date", "date"],
        "code": ["Code", "code"],
        "open": ["Open", "open"],
        "high": ["High", "high"],
        "low": ["Low", "low"],
        "close": ["Close", "close"],
        "adjusted": ["Adj.Close", "adjusted"],
        "volume": ["Volume", "volume"],
    }

    for std_name, aliases in col_map.items():
        for alias in aliases:
            if alias in df.columns and std_name not in df.columns:
                df = df.rename(columns={alias: std_name})
                break

    # ── 2. Extract trading dates ─────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"])

    if cfg.start_date:
        df = df[df["date"] >= pd.Timestamp(cfg.start_date)]
    if cfg.end_date:
        df = df[df["date"] <= pd.Timestamp(cfg.end_date)]

    dates = sorted(df["date"].unique())
    codes = sorted(df["code"].unique())

    if len(dates) == 0:
        raise ValueError("No valid data in the specified date range")

    # ── 3. Initialize state ──────────────────────────────────────────────
    cash = cfg.init_capital
    positions_long = {code: 0.0 for code in codes}   # shares held
    hold_cost = {code: 0.0 for code in codes}         # avg cost basis per share
    hold_high_water = {code: cfg.init_capital for code in codes}

    equity_curve: list[dict] = []
    daily_positions: list[dict] = []
    transactions: list[dict] = []

    # Rebalance day detection
    rebalance_dates = _get_rebalance_dates(dates, cfg)

    # ── 4. Day-by-day loop ───────────────────────────────────────────────
    for i, date in enumerate(dates):
        day_data = df[df["date"] == date].copy()
        if day_data.empty:
            continue

        is_rebalance = date in rebalance_dates
        nav_long = cash
        long_positions = {}

        for _, row in day_data.iterrows():
            code = row["code"]
            price_eval = row.get(cfg.eval_price_col, row.get("close", np.nan))
            price_exec = row.get(cfg.exec_price_col, row.get("open", np.nan))

            if np.isnan(price_eval) or np.isnan(price_exec):
                if cfg.skip_suspended:
                    continue

            # Mark-to-market long positions
            pos_val = positions_long.get(code, 0.0) * price_eval
            nav_long += pos_val
            long_positions[code] = {
                "shares": positions_long.get(code, 0.0),
                "price": price_eval,
                "value": pos_val,
            }

        portfolio_nav = nav_long

        # ── Check portfolio-level stops ──────────────────────────────────
        if i > 0 and len(equity_curve) > 0:
            prev_nav = equity_curve[-1]["nav"]
            drawdown_pct = (portfolio_nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0

            if cfg.enable_portfolio_stop_loss and _check_stop(
                portfolio_nav, equity_curve, cfg, level="portfolio", direction="loss"
            ):
                # Liquidate all positions
                for code in codes:
                    if positions_long.get(code, 0) > 0:
                        price = day_data.loc[day_data["code"] == code, cfg.exec_price_col]
                        if len(price) > 0:
                            p = price.iloc[0]
                            if not np.isnan(p):
                                proceeds = positions_long[code] * p * (1 - cfg.fee_rate - cfg.stamp_tax - cfg.slippage_rate)
                                cash += proceeds
                                transactions.append({
                                    "date": date, "code": code, "action": "SELL_STOP",
                                    "shares": positions_long[code], "price": p, "proceeds": proceeds,
                                })
                                positions_long[code] = 0.0
                portfolio_nav = cash

        # ── Rebalance ────────────────────────────────────────────────────
        if is_rebalance:
            target_nav = portfolio_nav * cfg.global_max_hold_pct
            target_pool = target_nav

            # Build list of (code, target_weight)
            targets = []
            for _, row in day_data.iterrows():
                code = row["code"]
                w = row.get(weight_col, 0.0)
                if np.isnan(w) or w < cfg.min_weight:
                    continue
                if w > cfg.single_max_weight:
                    w = cfg.single_max_weight
                targets.append((code, w))

            if targets:
                # Normalize weights to sum ≤ 1
                total_w = sum(w for _, w in targets)
                if total_w > 0:
                    for code, w in targets:
                        target_value = target_pool * (w / total_w)
                        price = day_data.loc[day_data["code"] == code, cfg.exec_price_col]
                        if len(price) == 0:
                            continue
                        p = price.iloc[0]
                        if np.isnan(p) or p <= 0:
                            continue

                        target_shares = int(target_value / p / cfg.lot_size) * cfg.lot_size
                        current_shares = positions_long.get(code, 0.0)
                        diff = target_shares - current_shares

                        if diff > 0:
                            # Buy
                            cost = diff * p * (1 + cfg.fee_rate + cfg.slippage_rate)
                            if cost <= cash:
                                cash -= cost
                                positions_long[code] = target_shares
                                if target_shares > 0:
                                    hold_cost[code] = (
                                        (hold_cost[code] * current_shares + p * diff) / target_shares
                                        if target_shares > 0 else p
                                    )
                                hold_high_water[code] = max(
                                    hold_high_water.get(code, 0), p
                                )
                                transactions.append({
                                    "date": date, "code": code, "action": "BUY",
                                    "shares": diff, "price": p, "cost": cost,
                                })
                        elif diff < 0:
                            # Sell
                            sell_shares = abs(diff)
                            proceeds = sell_shares * p * (1 - cfg.fee_rate - cfg.stamp_tax - cfg.slippage_rate)
                            cash += proceeds
                            positions_long[code] = target_shares
                            if target_shares == 0:
                                hold_cost[code] = 0.0
                                hold_high_water[code] = 0.0
                            transactions.append({
                                "date": date, "code": code, "action": "SELL",
                                "shares": sell_shares, "price": p, "proceeds": proceeds,
                            })

        # ── Component-level stop-loss / take-profit ──────────────────────
        if cfg.enable_component_stop_loss or cfg.enable_component_take_profit:
            for code in codes:
                if positions_long.get(code, 0) <= 0:
                    continue
                price_row = day_data.loc[day_data["code"] == code]
                if len(price_row) == 0:
                    continue
                p = price_row[cfg.exec_price_col].iloc[0] if cfg.exec_price_col in price_row.columns else price_row["close"].iloc[0]
                if np.isnan(p):
                    continue

                cost = hold_cost.get(code, p)
                hw = hold_high_water.get(code, p)

                # Stop loss check
                if cfg.enable_component_stop_loss:
                    sl_trigger = _get_stop_price(cost, hw, p, cfg, level="component", direction="loss")
                    if sl_trigger is not None and p <= sl_trigger:
                        proceeds = positions_long[code] * p * (1 - cfg.fee_rate - cfg.stamp_tax - cfg.slippage_rate)
                        cash += proceeds
                        transactions.append({
                            "date": date, "code": code, "action": "SELL_SL",
                            "shares": positions_long[code], "price": p, "proceeds": proceeds,
                        })
                        positions_long[code] = 0.0
                        continue

                # Take profit check
                if cfg.enable_component_take_profit:
                    tp_trigger = _get_stop_price(cost, hw, p, cfg, level="component", direction="profit")
                    if tp_trigger is not None and p >= tp_trigger:
                        proceeds = positions_long[code] * p * (1 - cfg.fee_rate - cfg.stamp_tax - cfg.slippage_rate)
                        cash += proceeds
                        transactions.append({
                            "date": date, "code": code, "action": "SELL_TP",
                            "shares": positions_long[code], "price": p, "proceeds": proceeds,
                        })
                        positions_long[code] = 0.0

                # Update high water mark
                if p > hw:
                    hold_high_water[code] = p

        # ── Record daily positions ───────────────────────────────────────
        for code in codes:
            price_row = day_data.loc[day_data["code"] == code]
            if len(price_row) == 0:
                continue
            pe = price_row.get(cfg.eval_price_col, price_row.get("close"))
            pe_val = pe.iloc[0] if len(pe) > 0 else np.nan
            sh = positions_long.get(code, 0.0)
            daily_positions.append({
                "date": date, "code": code,
                "shares": sh,
                "price": pe_val,
                "value": sh * pe_val if not np.isnan(pe_val) else 0.0,
            })

        # ── Record equity curve ──────────────────────────────────────────
        nav = cash + sum(
            positions_long.get(code, 0.0) * (
                day_data.loc[day_data["code"] == code, cfg.eval_price_col].iloc[0]
                if code in day_data["code"].values and cfg.eval_price_col in day_data.columns
                else day_data.loc[day_data["code"] == code, "close"].iloc[0]
                if code in day_data["code"].values
                else 0.0
            )
            for code in codes
            if positions_long.get(code, 0) > 0
        )

        daily_ret = (nav - equity_curve[-1]["nav"]) / equity_curve[-1]["nav"] if equity_curve else 0.0

        equity_curve.append({
            "date": date,
            "nav": nav,
            "cash": cash,
            "return": daily_ret,
        })

    # ── 5. Build result DataFrames ───────────────────────────────────────
    eq_df = pd.DataFrame(equity_curve)
    pos_df = pd.DataFrame(daily_positions)
    txn_df = pd.DataFrame(transactions)

    # Summary statistics
    if len(eq_df) > 1:
        total_return = (eq_df["nav"].iloc[-1] - cfg.init_capital) / cfg.init_capital
        daily_rets = eq_df["return"].dropna()
        sharpe = np.sqrt(252) * daily_rets.mean() / daily_rets.std() if daily_rets.std() > 0 else 0.0
        cummax = eq_df["nav"].cummax()
        drawdown = (eq_df["nav"] - cummax) / cummax
        max_dd = drawdown.min()
    else:
        total_return = 0.0
        sharpe = 0.0
        max_dd = 0.0

    summary = {
        "init_capital": cfg.init_capital,
        "final_nav": eq_df["nav"].iloc[-1] if len(eq_df) > 0 else cfg.init_capital,
        "total_return_pct": total_return * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "n_trades": len(txn_df),
        "n_days": len(eq_df),
    }

    return BacktestResult(
        daily_positions=pos_df,
        equity_curve=eq_df,
        transactions=txn_df,
        config=cfg.to_dict(),
        summary=summary,
    )


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════


def _get_rebalance_dates(dates: list, cfg: BacktestConfig) -> set:
    """Determine which dates are rebalance dates."""
    rebalance_dates = set()

    if cfg.rebalance_mode == "weight_shift":
        return set(dates)

    if str(cfg.rebalance_cycle).isdigit():
        n = int(cfg.rebalance_cycle)
        for i, d in enumerate(dates):
            if i % n == 0:
                rebalance_dates.add(d)
    else:
        cycle_map = {
            "daily": 1, "weekly": 5, "monthly": 21,
            "quarterly": 63, "semiannual": 126, "annual": 252,
        }
        n = cycle_map.get(cfg.rebalance_cycle, 63)
        for i, d in enumerate(dates):
            if i % n == 0:
                rebalance_dates.add(d)

    return rebalance_dates


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """Compute ATR for an asset series."""
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    tr_full = np.concatenate([[np.nan], tr])
    return pd.Series(tr_full).rolling(window=n, min_periods=n).mean().values


def _compute_volatility(price: np.ndarray, n: int) -> np.ndarray:
    """Compute rolling volatility (std of returns)."""
    rets = np.full(len(price), np.nan)
    rets[1:] = price[1:] / np.maximum(np.abs(price[:-1]), 1e-15) - 1.0
    return pd.Series(rets).rolling(window=n, min_periods=n).std(ddof=1).values


def _compute_log_vol(price: np.ndarray, n: int) -> np.ndarray:
    """Compute rolling log-return volatility."""
    rets = np.full(len(price), np.nan)
    rets[1:] = np.log(price[1:] / np.maximum(np.abs(price[:-1]), 1e-15))
    return pd.Series(rets).rolling(window=n, min_periods=n).std(ddof=1).values


def _get_stop_price_full(
    cost: float,
    high_water: float,
    current_price: float,
    atr_val: float,
    vol_val: float,
    log_vol_val: float,
    cfg: BacktestConfig,
    level: str = "component",
    direction: str = "loss",
) -> Optional[float]:
    """Calculate stop-loss or take-profit trigger price.

    Supports all 5 stop/take-profit types:
    - fixed: cost * (1 ± ratio)
    - trailing_fixed: high_water * (1 ± ratio)
    - trailing_atr: high_water ± k * ATR
    - trailing_vol: high_water ± sigma * vol * price
    - trailing_log: high_water ± sigma * log_vol * price

    Parameters
    ----------
    level : str
        ``"component"`` or ``"portfolio"``.
    direction : str
        ``"loss"`` (stop-loss, price goes DOWN) or ``"profit"`` (take-profit, price goes UP).
    """
    prefix = f"{level}_"

    if direction == "profit":
        st_type = getattr(cfg, f"{prefix}take_profit_type")
    else:
        st_type = getattr(cfg, f"{prefix}stop_loss_type")

    # Helper: get the trigger price above (profit) or below (loss) a reference
    def _trigger(ref: float, delta: float) -> float:
        return ref * (1 + delta) if direction == "profit" else ref * (1 - delta)

    if st_type == "fixed":
        if direction == "profit":
            ratio = getattr(cfg, f"fixed_{prefix}tp_ratio")
        else:
            ratio = getattr(cfg, f"fixed_{prefix}sl_ratio")
        return _trigger(cost, ratio)

    elif st_type == "trailing_fixed":
        if direction == "profit":
            ratio = getattr(cfg, f"trailing_fixed_{prefix}tp_ratio")
        else:
            ratio = getattr(cfg, f"trailing_fixed_{prefix}sl_ratio")
        return _trigger(high_water, ratio)

    elif st_type == "trailing_atr":
        if direction == "profit":
            k = getattr(cfg, f"atr_k_{prefix}tp")
        else:
            k = getattr(cfg, f"atr_k_{prefix}")
        if np.isnan(atr_val):
            return None
        delta = k * atr_val / current_price if current_price > 0 else 0.0
        return _trigger(high_water, delta)

    elif st_type == "trailing_vol":
        if direction == "profit":
            sigma = getattr(cfg, f"vol_sigma_{prefix}tp")
        else:
            sigma = getattr(cfg, f"vol_sigma_{prefix}")
        if np.isnan(vol_val):
            return None
        delta = sigma * vol_val
        return _trigger(high_water, delta)

    elif st_type == "trailing_log":
        if direction == "profit":
            sigma = getattr(cfg, f"log_vol_sigma_{prefix}tp")
        else:
            sigma = getattr(cfg, f"log_vol_sigma_{prefix}")
        if np.isnan(log_vol_val):
            return None
        delta = sigma * log_vol_val
        return _trigger(high_water, delta)

    return None


# Backward-compatible wrapper used by existing engine code
def _get_stop_price(
    cost: float, high_water: float, current_price: float,
    cfg: BacktestConfig, level: str = "component", direction: str = "loss",
) -> Optional[float]:
    return _get_stop_price_full(
        cost=cost, high_water=high_water, current_price=current_price,
        atr_val=np.nan, vol_val=np.nan, log_vol_val=np.nan,
        cfg=cfg, level=level, direction=direction,
    )


def _check_stop(
    portfolio_nav: float,
    equity_curve: list,
    cfg: BacktestConfig,
    level: str = "portfolio",
    direction: str = "loss",
) -> bool:
    """Check if portfolio-level stop is triggered."""
    prefix = f"{level}_"
    if direction == "loss":
        if not getattr(cfg, f"enable_{prefix}stop_loss"):
            return False
        st_type = getattr(cfg, f"{prefix}stop_loss_type")
        if st_type == "fixed" and len(equity_curve) > 0:
            ratio = getattr(cfg, f"fixed_{prefix}sl_ratio")
            peak = max(e["nav"] for e in equity_curve)
            return portfolio_nav < peak * (1 - ratio)
        elif st_type == "trailing_fixed" and len(equity_curve) > 0:
            ratio = getattr(cfg, f"trailing_fixed_{prefix}sl_ratio")
            peak = max(e["nav"] for e in equity_curve)
            return portfolio_nav < peak * (1 - ratio)
    elif direction == "profit":
        if not getattr(cfg, f"enable_{prefix}take_profit"):
            return False
        st_type = getattr(cfg, f"{prefix}take_profit_type")
        if st_type == "fixed" and len(equity_curve) > 0:
            ratio = getattr(cfg, f"fixed_{prefix}tp_ratio")
            init = cfg.init_capital
            return portfolio_nav > init * (1 + ratio)
    return False


def _precompute_indicators(
    df: pd.DataFrame, codes: list, cfg: BacktestConfig
) -> dict:
    """Precompute ATR, vol, log_vol for each asset if needed for stop/take-profit."""
    indicators = {code: {} for code in codes}

    need_atr = (
        (cfg.enable_component_stop_loss and cfg.component_stop_loss_type == "trailing_atr")
        or (cfg.enable_component_take_profit and cfg.component_take_profit_type == "trailing_atr")
    )
    need_vol = (
        (cfg.enable_component_stop_loss and cfg.component_stop_loss_type == "trailing_vol")
        or (cfg.enable_component_take_profit and cfg.component_take_profit_type == "trailing_vol")
    )
    need_log_vol = (
        (cfg.enable_component_stop_loss and cfg.component_stop_loss_type == "trailing_log")
        or (cfg.enable_component_take_profit and cfg.component_take_profit_type == "trailing_log")
    )

    if not (need_atr or need_vol or need_log_vol):
        return indicators

    for code in codes:
        sub = df[df["code"] == code].sort_values("date")
        h = sub["high"].values.astype(np.float64) if "high" in sub.columns else sub["close"].values
        l = sub["low"].values.astype(np.float64) if "low" in sub.columns else sub["close"].values
        c = sub["close"].values.astype(np.float64)

        if need_atr:
            indicators[code]["atr"] = _compute_atr(
                h, l, c,
                cfg.atr_n_component,
            )
        if need_vol:
            indicators[code]["vol"] = _compute_volatility(
                c, cfg.vol_n_component,
            )
        if need_log_vol:
            indicators[code]["log_vol"] = _compute_log_vol(
                c, cfg.log_vol_n_component,
            )

        # Map date index for fast lookup
        date_to_idx = {d: i for i, d in enumerate(sub["date"].values)}
        indicators[code]["_date_idx"] = date_to_idx

    return indicators
