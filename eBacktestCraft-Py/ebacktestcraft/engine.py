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

    return _run_engine(df.copy(), config, weight_col)


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

    # ── 3. Pre-compute technical indicators (ATR/vol/log_vol for stops) ──
    indicators = _precompute_indicators(df, codes, cfg)

    # ── 4. Build calendar rebalance set ─────────────────────────────────
    calendar_dates = _get_calendar_rebalance_dates(dates, cfg)

    # ── 5. Build weight-shift index for hybrid/weight_shift detection ────
    # Pre-index weight by (date, code) for O(1) lookup during the loop
    weight_index: dict = {}
    if cfg.rebalance_mode in ("weight_shift", "hybrid"):
        for _, row in df[["date", "code", weight_col]].iterrows():
            weight_index[(row["date"], row["code"])] = row[weight_col]

    # ── 6. Initialize portfolio state ───────────────────────────────────
    cash = float(cfg.init_capital)
    positions_long: dict[str, float] = {code: 0.0 for code in codes}
    hold_cost: dict[str, float] = {code: 0.0 for code in codes}
    # BUG FIX: initialize high-water to 0, not init_capital
    hold_high_water: dict[str, float] = {code: 0.0 for code in codes}
    stopped_in_cycle: dict[str, bool] = {code: False for code in codes}

    portfolio_high_water = float(cfg.init_capital)

    equity_curve: list[dict] = []
    daily_positions: list[dict] = []
    transactions: list[dict] = []

    # Previous weights per code for weight_shift detection
    prev_weights: dict[str, float] = {code: 0.0 for code in codes}

    # ── 7. Day-by-day loop ───────────────────────────────────────────────
    for i, date in enumerate(dates):
        day_data = df[df["date"] == date]
        if day_data.empty:
            continue

        # ── 7.1 Build price lookups for today ───────────────────────────
        exec_prices: dict[str, float] = {}
        eval_prices: dict[str, float] = {}
        for _, row in day_data.iterrows():
            code = row["code"]
            ep = row.get(cfg.exec_price_col, row.get("open", np.nan))
            vp = row.get(cfg.eval_price_col, row.get("close", np.nan))
            exec_prices[code] = float(ep) if not (ep is None or (isinstance(ep, float) and np.isnan(ep))) else np.nan
            eval_prices[code] = float(vp) if not (vp is None or (isinstance(vp, float) and np.isnan(vp))) else np.nan

        # ── 7.2 Mark-to-market NAV ──────────────────────────────────────
        portfolio_nav = cash
        for code in codes:
            sh = positions_long[code]
            if sh > 0:
                vp = eval_prices.get(code, np.nan)
                if not np.isnan(vp):
                    portfolio_nav += sh * vp

        portfolio_high_water = max(portfolio_high_water, portfolio_nav)

        # ── 7.3 Determine if this is a rebalance day ────────────────────
        is_calendar_rb = date in calendar_dates

        is_weight_shift_rb = False
        if cfg.rebalance_mode in ("weight_shift", "hybrid") and i > 0:
            for code in codes:
                new_w = weight_index.get((date, code), 0.0)
                if np.isnan(new_w):
                    new_w = 0.0
                if abs(new_w - prev_weights.get(code, 0.0)) > cfg.weight_change_threshold:
                    is_weight_shift_rb = True
                    break

        if cfg.rebalance_mode == "calendar":
            is_rebalance = is_calendar_rb
        elif cfg.rebalance_mode == "weight_shift":
            is_rebalance = is_weight_shift_rb or i == 0
        else:  # hybrid
            is_rebalance = is_calendar_rb or is_weight_shift_rb

        # First day always rebalances
        if i == 0:
            is_rebalance = True

        # ── 7.4 Portfolio-level stop-loss (before rebalance) ────────────
        pf_stopped = False
        check_pf_sl = cfg.enable_portfolio_stop_loss or cfg.enable_oco_portfolio
        if i > 0 and check_pf_sl:
            pf_stopped = _check_portfolio_stop(
                portfolio_nav, portfolio_high_water, cfg.init_capital,
                equity_curve, cfg, direction="loss",
            )
            if pf_stopped:
                for code in list(codes):
                    if positions_long[code] <= 0:
                        continue
                    ep = exec_prices.get(code, np.nan)
                    if np.isnan(ep) or ep <= 0:
                        continue
                    ep_slip = ep * (1 - cfg.slippage_rate)
                    proceeds = positions_long[code] * ep_slip
                    fee = proceeds * cfg.fee_rate
                    stax = proceeds * cfg.stamp_tax
                    cash += proceeds - fee - stax
                    transactions.append({
                        "date": date, "code": code, "action": "SELL_PORTFOLIO_SL",
                        "shares": positions_long[code], "price": ep,
                        "exec_price": ep_slip, "proceeds": proceeds - fee - stax,
                    })
                    positions_long[code] = 0.0
                    hold_cost[code] = 0.0
                    hold_high_water[code] = 0.0
                    stopped_in_cycle[code] = True
                is_rebalance = False  # skip rebalance when portfolio stopped

        # ── 7.5 Portfolio-level take-profit (before rebalance) ───────────
        pf_tp = False
        check_pf_tp = cfg.enable_portfolio_take_profit or cfg.enable_oco_portfolio
        if i > 0 and check_pf_tp and not pf_stopped:
            pf_tp = _check_portfolio_stop(
                portfolio_nav, portfolio_high_water, cfg.init_capital,
                equity_curve, cfg, direction="profit",
            )
            if pf_tp:
                for code in list(codes):
                    if positions_long[code] <= 0:
                        continue
                    ep = exec_prices.get(code, np.nan)
                    if np.isnan(ep) or ep <= 0:
                        continue
                    ep_slip = ep * (1 - cfg.slippage_rate)
                    proceeds = positions_long[code] * ep_slip
                    fee = proceeds * cfg.fee_rate
                    stax = proceeds * cfg.stamp_tax
                    cash += proceeds - fee - stax
                    transactions.append({
                        "date": date, "code": code, "action": "SELL_PORTFOLIO_TP",
                        "shares": positions_long[code], "price": ep,
                        "exec_price": ep_slip, "proceeds": proceeds - fee - stax,
                    })
                    positions_long[code] = 0.0
                    hold_cost[code] = 0.0
                    hold_high_water[code] = 0.0
                is_rebalance = False

        # ── 7.6 Component-level stop-loss / take-profit (before rebalance)
        check_component_sl = cfg.enable_component_stop_loss or cfg.enable_oco_component
        check_component_tp = cfg.enable_component_take_profit or cfg.enable_oco_component
        if not pf_stopped and not pf_tp and (check_component_sl or check_component_tp):
            for code in codes:
                if positions_long[code] <= 0:
                    continue
                ep = exec_prices.get(code, np.nan)
                if np.isnan(ep) or ep <= 0:
                    continue

                cost_basis = hold_cost[code]
                hw = hold_high_water[code]

                # Look up pre-computed indicator values for this date
                ind = indicators.get(code, {})
                date_idx_map = ind.get("_date_idx", {})
                didx = date_idx_map.get(date, None)
                atr_val = ind["atr"][didx] if didx is not None and "atr" in ind else np.nan
                vol_val = ind["vol"][didx] if didx is not None and "vol" in ind else np.nan
                log_vol_val = ind["log_vol"][didx] if didx is not None and "log_vol" in ind else np.nan

                sold = False

                # ── Stop-loss check ──────────────────────────────────────
                if check_component_sl:
                    sl_price = _get_stop_price_full(
                        cost_basis, hw, ep, atr_val, vol_val, log_vol_val,
                        cfg, level="component", direction="loss",
                    )
                    triggered = sl_price is not None and ep <= sl_price
                    # stop_limit: also verify exec price didn't gap through limit floor
                    if triggered and cfg.component_stop_loss_type == "stop_limit":
                        limit_floor = sl_price * (1 - cfg.stop_limit_component_sl_gap)
                        triggered = ep >= limit_floor  # no fill if gapped below floor
                    if triggered:
                        # Limit orders fill at exec price (no extra slippage vs. market)
                        is_limit = cfg.component_stop_loss_type == "stop_limit"
                        ep_exec = ep if is_limit else ep * (1 - cfg.slippage_rate)
                        proceeds = positions_long[code] * ep_exec
                        fee = proceeds * cfg.fee_rate
                        stax = proceeds * cfg.stamp_tax
                        cash += proceeds - fee - stax
                        transactions.append({
                            "date": date, "code": code,
                            "action": "SELL_SL_LIMIT" if is_limit else "SELL_SL",
                            "shares": positions_long[code], "price": ep,
                            "exec_price": ep_exec, "proceeds": proceeds - fee - stax,
                        })
                        positions_long[code] = 0.0
                        hold_cost[code] = 0.0
                        hold_high_water[code] = 0.0
                        stopped_in_cycle[code] = True
                        sold = True

                # ── Take-profit check (OCO: only if SL didn't fire) ──────
                if not sold and check_component_tp:
                    tp_price = _get_stop_price_full(
                        cost_basis, hw, ep, atr_val, vol_val, log_vol_val,
                        cfg, level="component", direction="profit",
                    )
                    triggered = tp_price is not None and ep >= tp_price
                    # stop_limit TP: verify price didn't gap above limit ceiling
                    if triggered and cfg.component_take_profit_type == "stop_limit":
                        limit_ceil = tp_price * (1 + cfg.stop_limit_component_tp_gap)
                        triggered = ep <= limit_ceil  # no fill if gapped above ceiling
                    if triggered:
                        is_limit = cfg.component_take_profit_type == "stop_limit"
                        ep_exec = ep if is_limit else ep * (1 - cfg.slippage_rate)
                        proceeds = positions_long[code] * ep_exec
                        fee = proceeds * cfg.fee_rate
                        stax = proceeds * cfg.stamp_tax
                        cash += proceeds - fee - stax
                        transactions.append({
                            "date": date, "code": code,
                            "action": "SELL_TP_LIMIT" if is_limit else "SELL_TP",
                            "shares": positions_long[code], "price": ep,
                            "exec_price": ep_exec, "proceeds": proceeds - fee - stax,
                        })
                        positions_long[code] = 0.0
                        hold_cost[code] = 0.0
                        hold_high_water[code] = 0.0
                        sold = True

                if not sold and ep > hw:
                    hold_high_water[code] = ep

        # ── 7.7 Rebalance: sell excess first, then buy ───────────────────
        if is_rebalance:
            # Reset stopped_in_cycle for new rebalance window
            for code in codes:
                stopped_in_cycle[code] = False

            total_asset = cash + sum(
                positions_long[c] * eval_prices.get(c, 0.0)
                for c in codes
                if positions_long[c] > 0 and not np.isnan(eval_prices.get(c, np.nan))
            )

            # Build target weights, cap and clip
            target_w: dict[str, float] = {}
            for _, row in day_data.iterrows():
                code = row["code"]
                w = row.get(weight_col, 0.0)
                if w is None or (isinstance(w, float) and np.isnan(w)):
                    w = 0.0
                w = min(float(w), cfg.single_max_weight)
                if w >= cfg.min_weight:
                    target_w[code] = w

            # Normalize to sum ≤ global_max_hold_pct
            total_w = sum(target_w.values())
            if total_w > 0:
                scale = cfg.global_max_hold_pct / total_w
                target_w = {c: w * scale for c, w in target_w.items()}

            # Compute target shares per code
            target_shares: dict[str, float] = {}
            for code in codes:
                ep = exec_prices.get(code, np.nan)
                w = target_w.get(code, 0.0)
                if np.isnan(ep) or ep <= 0 or w <= 0:
                    target_shares[code] = 0.0
                else:
                    raw = total_asset * w / ep
                    target_shares[code] = float(int(raw / cfg.lot_size) * cfg.lot_size)

            # Sell excess first (frees cash)
            for code in codes:
                current = positions_long[code]
                target = target_shares[code]
                if current > target:
                    sell_q = current - target
                    ep = exec_prices.get(code, np.nan)
                    if np.isnan(ep) or ep <= 0:
                        continue
                    ep_slip = ep * (1 - cfg.slippage_rate)
                    proceeds = sell_q * ep_slip
                    fee = proceeds * cfg.fee_rate
                    stax = proceeds * cfg.stamp_tax
                    cash += proceeds - fee - stax
                    positions_long[code] = target
                    if target == 0:
                        hold_cost[code] = 0.0
                        hold_high_water[code] = 0.0
                    transactions.append({
                        "date": date, "code": code, "action": "SELL",
                        "shares": sell_q, "price": ep,
                        "exec_price": ep_slip, "proceeds": proceeds - fee - stax,
                    })

            # Then buy shortfalls
            for code in codes:
                current = positions_long[code]
                target = target_shares[code]
                if target > current:
                    buy_q = target - current
                    ep = exec_prices.get(code, np.nan)
                    if np.isnan(ep) or ep <= 0:
                        continue
                    ep_slip = ep * (1 + cfg.slippage_rate)
                    cost_buy = buy_q * ep_slip
                    fee = cost_buy * cfg.fee_rate
                    total_cost = cost_buy + fee
                    if total_cost > cash:
                        continue
                    cash -= total_cost
                    old_sh = positions_long[code]
                    positions_long[code] = target
                    if old_sh == 0:
                        hold_cost[code] = ep_slip
                        hold_high_water[code] = ep
                    else:
                        hold_cost[code] = (hold_cost[code] * old_sh + ep_slip * buy_q) / target
                        hold_high_water[code] = max(hold_high_water[code], ep)
                    transactions.append({
                        "date": date, "code": code, "action": "BUY",
                        "shares": buy_q, "price": ep,
                        "exec_price": ep_slip, "cost": total_cost,
                    })

            # Update prev_weights for next-day weight_shift detection
            for code in codes:
                w = weight_index.get((date, code), 0.0)
                prev_weights[code] = 0.0 if (w is None or np.isnan(w)) else float(w)

        # ── 7.8 Record daily positions ───────────────────────────────────
        for code in codes:
            vp = eval_prices.get(code, np.nan)
            sh = positions_long[code]
            daily_positions.append({
                "date": date, "code": code,
                "shares": sh,
                "eval_price": vp,
                "market_value": sh * vp if not np.isnan(vp) else 0.0,
                "is_rebalance_day": is_rebalance,
            })

        # ── 7.9 Record equity curve ──────────────────────────────────────
        nav = cash
        for code in codes:
            if positions_long[code] > 0:
                vp = eval_prices.get(code, np.nan)
                if not np.isnan(vp):
                    nav += positions_long[code] * vp

        daily_ret = (nav / equity_curve[-1]["nav"] - 1.0) if equity_curve else 0.0

        equity_curve.append({
            "date": date,
            "nav": nav,
            "cash": cash,
            "return": daily_ret,
            "is_rebalance_day": is_rebalance,
            "portfolio_high_water": portfolio_high_water,
        })

    # ── 8. Build result DataFrames ───────────────────────────────────────
    eq_df = pd.DataFrame(equity_curve)
    pos_df = pd.DataFrame(daily_positions)
    txn_df = pd.DataFrame(transactions)

    if len(eq_df) > 1:
        total_return = (eq_df["nav"].iloc[-1] - cfg.init_capital) / cfg.init_capital
        daily_rets = eq_df["return"].dropna()
        vol = daily_rets.std()
        sharpe = float(np.sqrt(252) * daily_rets.mean() / vol) if vol > 0 else 0.0
        cummax = eq_df["nav"].cummax()
        drawdown = (eq_df["nav"] - cummax) / cummax
        max_dd = float(drawdown.min())
        n_days = len(eq_df)
        annual_ret = float((eq_df["nav"].iloc[-1] / cfg.init_capital) ** (252 / n_days) - 1)
    else:
        total_return = annual_ret = sharpe = max_dd = 0.0
        n_days = len(eq_df)

    summary = {
        "init_capital": cfg.init_capital,
        "final_nav": float(eq_df["nav"].iloc[-1]) if len(eq_df) > 0 else cfg.init_capital,
        "total_return_pct": total_return * 100,
        "annual_return_pct": annual_ret * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "n_trades": len(txn_df),
        "n_days": n_days,
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


def _get_calendar_rebalance_dates(dates: list, cfg: BacktestConfig) -> set:
    """Return the set of calendar-triggered rebalance dates.

    Uses true calendar periods (first trading day of each period) rather
    than fixed trading-day counts.
    """
    rebalance_dates: set = set()

    cycle = str(cfg.rebalance_cycle)

    # Numeric cycle: every N trading days
    if cycle.isdigit():
        n = int(cycle)
        for i, d in enumerate(dates):
            if i % n == 0:
                rebalance_dates.add(d)
        return rebalance_dates

    # Named cycle: use pandas period grouping for true calendar alignment
    # Period-compatible aliases (work across pandas versions)
    freq_map = {
        "daily": "D",
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
        "semiannual": "6M",
        "annual": "A",
    }
    freq = freq_map.get(cycle)
    if freq is None:
        raise ValueError(
            f"rebalance_cycle must be a positive integer or one of: "
            f"{', '.join(freq_map)}"
        )

    date_series = pd.Series(dates)
    periods = date_series.dt.to_period(freq)
    # First trading day in each calendar period
    first_per_period = date_series.groupby(periods).first()
    rebalance_dates = set(first_per_period.values)
    return rebalance_dates


def _check_portfolio_stop(
    portfolio_nav: float,
    portfolio_high_water: float,
    init_capital: float,
    equity_curve: list,
    cfg: BacktestConfig,
    direction: str = "loss",
) -> bool:
    """Check portfolio-level stop-loss or take-profit trigger."""
    if direction == "loss":
        st_type = cfg.portfolio_stop_loss_type
        if st_type == "fixed":
            ratio = cfg.fixed_portfolio_sl_ratio
            return portfolio_nav < init_capital * (1 - ratio)
        elif st_type in ("trailing_fixed", "stop_limit"):
            ratio = cfg.trailing_fixed_portfolio_sl_ratio
            trigger = portfolio_high_water * (1 - ratio)
            triggered = portfolio_nav < trigger
            if triggered and st_type == "stop_limit":
                # Portfolio stop_limit: don't fire if NAV gapped way below floor
                floor = trigger * (1 - cfg.stop_limit_portfolio_sl_gap)
                triggered = portfolio_nav >= floor
            return triggered
        elif st_type in ("trailing_atr", "trailing_vol", "trailing_log") and len(equity_curve) > 0:
            navs = np.array([e["nav"] for e in equity_curve])
            if st_type == "trailing_atr":
                n = cfg.atr_n_portfolio
                diffs = np.diff(navs)
                atr = np.mean(np.abs(diffs[-n:])) if len(diffs) >= n else np.nan
                if np.isnan(atr):
                    return False
                return portfolio_nav < portfolio_high_water - cfg.atr_k_portfolio * atr
            elif st_type == "trailing_vol":
                n = cfg.vol_n_portfolio
                rets = np.diff(navs) / navs[:-1]
                vol = np.std(rets[-n:], ddof=1) if len(rets) >= n else np.nan
                if np.isnan(vol):
                    return False
                return portfolio_nav < portfolio_high_water * (1 - cfg.vol_sigma_portfolio * vol)
            elif st_type == "trailing_log":
                n = cfg.log_vol_n_portfolio
                log_rets = np.diff(np.log(np.maximum(navs, 1e-10)))
                lvol = np.std(log_rets[-n:], ddof=1) if len(log_rets) >= n else np.nan
                if np.isnan(lvol):
                    return False
                return portfolio_nav < portfolio_high_water * np.exp(-cfg.log_vol_sigma_portfolio * lvol)
    else:  # profit
        st_type = cfg.portfolio_take_profit_type
        if st_type == "fixed":
            ratio = cfg.fixed_portfolio_tp_ratio
            return portfolio_nav > init_capital * (1 + ratio)
        elif st_type in ("trailing_fixed", "stop_limit"):
            ratio = cfg.trailing_fixed_portfolio_tp_ratio
            trigger = portfolio_high_water * (1 - ratio)
            triggered = portfolio_nav < trigger
            if triggered and st_type == "stop_limit":
                ceil = trigger * (1 + cfg.stop_limit_portfolio_tp_gap)
                triggered = portfolio_nav <= ceil
            return triggered
        elif st_type in ("trailing_atr", "trailing_vol", "trailing_log") and len(equity_curve) > 0:
            navs = np.array([e["nav"] for e in equity_curve])
            if st_type == "trailing_atr":
                n = cfg.atr_n_portfolio
                diffs = np.diff(navs)
                atr = np.mean(np.abs(diffs[-n:])) if len(diffs) >= n else np.nan
                if np.isnan(atr):
                    return False
                return portfolio_nav < portfolio_high_water - cfg.atr_k_portfolio_tp * atr
            elif st_type == "trailing_vol":
                n = cfg.vol_n_portfolio
                rets = np.diff(navs) / navs[:-1]
                vol = np.std(rets[-n:], ddof=1) if len(rets) >= n else np.nan
                if np.isnan(vol):
                    return False
                return portfolio_nav < portfolio_high_water * (1 - cfg.vol_sigma_portfolio_tp * vol)
            elif st_type == "trailing_log":
                n = cfg.log_vol_n_portfolio
                log_rets = np.diff(np.log(np.maximum(navs, 1e-10)))
                lvol = np.std(log_rets[-n:], ddof=1) if len(log_rets) >= n else np.nan
                if np.isnan(lvol):
                    return False
                return portfolio_nav < portfolio_high_water * np.exp(-cfg.log_vol_sigma_portfolio_tp * lvol)
    return False


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
    """Calculate component stop-loss or take-profit trigger price."""
    lvl = level  # "component" or "portfolio"

    if direction == "profit":
        st_type = getattr(cfg, f"{lvl}_take_profit_type")
    else:
        st_type = getattr(cfg, f"{lvl}_stop_loss_type")

    def _trigger(ref: float, delta: float) -> float:
        return ref * (1 + delta) if direction == "profit" else ref * (1 - delta)

    if st_type == "fixed":
        attr = f"fixed_{lvl}_tp_ratio" if direction == "profit" else f"fixed_{lvl}_sl_ratio"
        return _trigger(cost, getattr(cfg, attr))

    elif st_type == "trailing_fixed":
        attr = f"trailing_fixed_{lvl}_tp_ratio" if direction == "profit" else f"trailing_fixed_{lvl}_sl_ratio"
        return _trigger(high_water, getattr(cfg, attr))

    elif st_type == "trailing_atr":
        if np.isnan(atr_val):
            return None
        # config attrs: atr_k_component / atr_k_component_tp  (no trailing _)
        lvl = level  # "component" or "portfolio"
        k_attr = f"atr_k_{lvl}_tp" if direction == "profit" else f"atr_k_{lvl}"
        k = getattr(cfg, k_attr)
        delta = k * atr_val / current_price if current_price > 0 else 0.0
        return _trigger(high_water, delta)

    elif st_type == "trailing_vol":
        if np.isnan(vol_val):
            return None
        lvl = level
        s_attr = f"vol_sigma_{lvl}_tp" if direction == "profit" else f"vol_sigma_{lvl}"
        return _trigger(high_water, getattr(cfg, s_attr) * vol_val)

    elif st_type == "trailing_log":
        if np.isnan(log_vol_val):
            return None
        lvl = level
        s_attr = f"log_vol_sigma_{lvl}_tp" if direction == "profit" else f"log_vol_sigma_{lvl}"
        return _trigger(high_water, getattr(cfg, s_attr) * log_vol_val)

    elif st_type == "stop_limit":
        # Stop-limit: trigger at trailing_fixed level; fills only if exec price
        # stays within the limit gap (prevents fills on extreme gap-through moves).
        # Returns the trigger price; gap check is done in the engine loop.
        lvl = level
        if direction == "loss":
            ratio = getattr(cfg, f"trailing_fixed_{lvl}_sl_ratio")
            return high_water * (1 - ratio)
        else:
            ratio = getattr(cfg, f"fixed_{lvl}_tp_ratio")
            return cost * (1 + ratio)

    return None


def _precompute_indicators(
    df: pd.DataFrame, codes: list, cfg: BacktestConfig
) -> dict:
    """Precompute ATR, vol, log_vol for each asset if needed for stop/take-profit."""
    indicators: dict = {code: {} for code in codes}

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
        sub = df[df["code"] == code].sort_values("date").reset_index(drop=True)
        h = sub["high"].values.astype(np.float64) if "high" in sub.columns else sub["close"].values.astype(np.float64)
        l = sub["low"].values.astype(np.float64) if "low" in sub.columns else sub["close"].values.astype(np.float64)
        c = sub["close"].values.astype(np.float64)

        if need_atr:
            indicators[code]["atr"] = _compute_atr(h, l, c, cfg.atr_n_component)
        if need_vol:
            indicators[code]["vol"] = _compute_volatility(c, cfg.vol_n_component)
        if need_log_vol:
            indicators[code]["log_vol"] = _compute_log_vol(c, cfg.log_vol_n_component)

        indicators[code]["_date_idx"] = {
            pd.Timestamp(d): idx for idx, d in enumerate(sub["date"].values)
        }

    return indicators


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    tr_full = np.concatenate([[np.nan], tr])
    return pd.Series(tr_full).rolling(window=n, min_periods=n).mean().values


def _compute_volatility(price: np.ndarray, n: int) -> np.ndarray:
    rets = np.full(len(price), np.nan)
    rets[1:] = price[1:] / np.maximum(np.abs(price[:-1]), 1e-15) - 1.0
    return pd.Series(rets).rolling(window=n, min_periods=n).std(ddof=1).values


def _compute_log_vol(price: np.ndarray, n: int) -> np.ndarray:
    rets = np.full(len(price), np.nan)
    rets[1:] = np.log(price[1:] / np.maximum(np.abs(price[:-1]), 1e-15))
    return pd.Series(rets).rolling(window=n, min_periods=n).std(ddof=1).values
