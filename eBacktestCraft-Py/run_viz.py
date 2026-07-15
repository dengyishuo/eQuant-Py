#!/usr/bin/env python3
"""Generate all quantkit visualization plots and save as PNG files."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import sys
sys.path.insert(0, "/Users/tool/Desktop")

import numpy as np
import pandas as pd
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# 1. Build mock panel data (better data: 8 stocks × 252 days, 1yr)
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
DATES = pd.date_range("2024-01-02", periods=252, freq="B")
CODES = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "BRK.B"]
NAMES = [
    "Apple Inc.", "Microsoft Corp.", "Alphabet Inc.", "Amazon.com Inc.",
    "Meta Platforms Inc.", "NVIDIA Corp.", "Tesla Inc.", "Berkshire Hathaway",
]

base_prices = {
    "AAPL": 185.0, "MSFT": 375.0, "GOOG": 140.0, "AMZN": 150.0,
    "META": 355.0, "NVDA": 490.0, "TSLA": 250.0, "BRK.B": 360.0,
}
trends = {
    "AAPL": 0.12, "MSFT": 0.15, "GOOG": 0.08, "AMZN": 0.18,
    "META": 0.22, "NVDA": 0.50, "TSLA": -0.05, "BRK.B": 0.10,
}
vols = {k: v * 0.02 for k, v in base_prices.items()}

rows = []
for d in DATES:
    day_idx = (d - DATES[0]).days
    for code in CODES:
        bp = base_prices[code]
        tr = trends[code]
        vol = vols[code]
        close = bp + tr * day_idx + np.random.normal(0, vol)
        if close < 10:
            close = 10.0
        open_p = close * np.random.uniform(0.995, 1.005)
        high = max(close, open_p) * np.random.uniform(1.002, 1.025)
        low = min(close, open_p) * np.random.uniform(0.975, 0.998)
        adj = close * np.random.uniform(0.998, 1.002)
        vol_qty = np.random.uniform(5e6, 8e7)
        cap = close * np.random.uniform(5e8, 3e9)
        bv = cap * np.random.uniform(0.1, 0.9)

        rows.append({
            "date": d, "code": code, "name": NAMES[CODES.index(code)],
            "open": round(open_p, 2), "high": round(high, 2), "low": round(low, 2),
            "close": round(close, 2), "adjusted": round(adj, 2),
            "volume": int(vol_qty), "cap": round(cap, 2), "bv": round(bv, 2),
        })

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])

print(f"Data: {len(CODES)} stocks × {len(DATES)} days = {len(df)} rows")
print(f"Date range: {DATES[0].date()} ~ {DATES[-1].date()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Factor pipeline
# ══════════════════════════════════════════════════════════════════════════════
from quantkit import factors as factor
from quantkit import indicators
from quantkit import engineering as eng
from quantkit import backtest

# Compute factors
df = factor.momentum(df, close_col="adjusted", n=[5, 10, 20, 60])
df = factor.value(df, bv_col="bv", cap_col="cap")
df = factor.size(df, cap_col="cap")
df = factor.volatility(df, close_col="adjusted", n=20)
df = factor.rps(df, close_col="adjusted", n=60)
df = factor.ram(df, close_col="adjusted", n=60, risk="vol")

# Technical indicators
df = indicators.rsi(df, n=14)
df = indicators.macd(df)
df = indicators.bollinger(df, n=20, sd=2.0)
df = indicators.atr(df, n=14)

print("Factors & indicators computed.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Factor engineering — IC analysis
# ══════════════════════════════════════════════════════════════════════════════
df = eng.add_next_return(df, close_col="adjusted", periods=[1, 5, 20])
df = eng.winsorize(df, factor_col="mom_20", probs=(0.01, 0.99))
df = eng.standardize(df, factor_col="mom_20")

ic = eng.ic_analysis(df, factor_cols=["mom_20", "mom_60", "rps_60", "ram_60"])
print("\nIC Analysis:")
for fname, ic_df in ic.items():
    fwd_cols = [c for c in ic_df.columns if c.startswith("forward")]
    for fc in fwd_cols:
        vals = ic_df[fc].dropna()
        ir = vals.mean() / vals.std() if vals.std() > 0 else 0
        print(f"  {fname} vs {fc}: IC_mean={vals.mean():.4f}, IC_std={vals.std():.4f}, IR={ir:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Generate signals
# ══════════════════════════════════════════════════════════════════════════════
# Primary strategy: momentum > 0
df = backtest.signal(
    df, indicator_cols=["mom_20"], signal_type="threshold",
    threshold=0, compare_op=">",
)
SIG_COL = "signal_mom_20_gt_0"
df = backtest.equal_weight(df, signal_col=SIG_COL)
WT_COL = f"weight_equal_{SIG_COL}"

# ══════════════════════════════════════════════════════════════════════════════
# 5. Run backtest
# ══════════════════════════════════════════════════════════════════════════════
cfg = backtest.Config(
    init_capital=1_000_000,
    rebalance_cycle="monthly",
    fee_rate=0.0003,
    stamp_tax=0.0,
    slippage_rate=0.001,
    lot_size=1,
    rebalance_mode="calendar",
    single_max_weight=0.40,
)
result = backtest.run(df, config=cfg, weight_col=WT_COL)

# ══════════════════════════════════════════════════════════════════════════════
# 6. Benchmark comparison
# ══════════════════════════════════════════════════════════════════════════════
bm_cfg = backtest.Config(init_capital=1_000_000, rebalance_cycle="monthly")
bm_equal = backtest.equal_weight_benchmark(df, bm_cfg)
bm_bh = backtest.buy_and_hold_benchmark(df, bm_cfg)

eq = result.equity_curve
metrics = backtest.performance_analysis(eq, init_capital=cfg.init_capital)

print(f"\nPerformance:")
print(f"  Total Return:   {metrics['total_return_pct']:+.2f}%")
print(f"  Annual Return:  {metrics['annual_return_pct']:+.2f}%")
print(f"  Annual Vol:     {metrics['annual_volatility_pct']:.2f}%")
print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
print(f"  Sortino Ratio:  {metrics['sortino_ratio']:.2f}")
print(f"  Calmar Ratio:   {metrics['calmar_ratio']:.2f}")
print(f"  Max Drawdown:   {metrics['max_drawdown_pct']:.2f}%")
print(f"  Win Rate:       {metrics['win_rate_pct']:.2f}%")
print(f"  Best Day:       {metrics['best_day_pct']:+.2f}%")
print(f"  Worst Day:      {metrics['worst_day_pct']:+.2f}%")
print(f"  Trades:         {metrics['n_trades']}")
print(f"  Final NAV:      ${metrics['final_nav']:,.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Generate all plots
# ══════════════════════════════════════════════════════════════════════════════
import os
OUT_DIR = "/Users/tool/Desktop/quantkit/plots"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"\nGenerating plots → {OUT_DIR}/")

# 7a. Equity curve with benchmarks
fig1 = backtest.plot_benchmark_compare(
    {"Strategy (mom_20)": eq,
     "Equal Weight": bm_equal.equity_curve,
     "Buy & Hold": bm_bh.equity_curve},
    title="Strategy vs Benchmarks — Momentum Factor",
)
fig1.savefig(f"{OUT_DIR}/01_benchmark_compare.png", dpi=150, bbox_inches="tight")
print("  ✓ 01_benchmark_compare.png")

# 7b. Equity curve (standalone)
fig2 = backtest.plot_equity_curve(
    eq, benchmark_curve=bm_bh.equity_curve,
    benchmark_label="Buy & Hold",
    title="Equity Curve — Momentum Strategy vs Buy & Hold",
)
fig2.savefig(f"{OUT_DIR}/02_equity_curve.png", dpi=150, bbox_inches="tight")
print("  ✓ 02_equity_curve.png")

# 7c. Drawdown
fig3 = backtest.plot_drawdown(eq, title="Drawdown — Momentum Strategy")
fig3.savefig(f"{OUT_DIR}/03_drawdown.png", dpi=150, bbox_inches="tight")
print("  ✓ 03_drawdown.png")

# 7d. Combined return + drawdown
fig4 = backtest.plot_return_drawdown(eq, title="Performance — Momentum Strategy")
fig4.savefig(f"{OUT_DIR}/04_performance.png", dpi=150, bbox_inches="tight")
print("  ✓ 04_performance.png")

# 7e. Return distribution
fig5 = backtest.plot_return_dist(eq, bins=40, title="Daily Return Distribution — Momentum Strategy")
fig5.savefig(f"{OUT_DIR}/05_return_dist.png", dpi=150, bbox_inches="tight")
print("  ✓ 05_return_dist.png")

# 7f. Monthly returns heatmap
fig6 = backtest.plot_monthly_return(eq, title="Monthly Returns (%) — Momentum Strategy")
fig6.savefig(f"{OUT_DIR}/06_monthly_heatmap.png", dpi=150, bbox_inches="tight")
print("  ✓ 06_monthly_heatmap.png")

# 7g. plot_all (generates 5 standard plots)
all_figs = backtest.plot_all(eq, title_prefix="Momentum (mom_20)", save_dir=OUT_DIR)
for name in all_figs:
    fname = f"{OUT_DIR}/{name}.png"
    all_figs[name].savefig(fname, dpi=150, bbox_inches="tight")
print("  ✓ plot_all complete")

print(f"\nAll plots saved to {OUT_DIR}/")
print("Done!")
