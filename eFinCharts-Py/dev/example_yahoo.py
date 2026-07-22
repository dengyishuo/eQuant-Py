"""eFinCharts example — Get data from Yahoo Finance and plot."""

import yfinance as yf
from efincharts import candlestick

# ---------------------------------------------------------------------------
# Get data
# ---------------------------------------------------------------------------
print("Fetching data from Yahoo Finance...")
df = yf.download("AAPL", start="2025-01-01", end="2026-07-22", auto_adjust=True)
print(f"{len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")

# ---------------------------------------------------------------------------
# Example 1: Basic candlestick + volume + MA
# ---------------------------------------------------------------------------
print("\n[1/4] Basic: candlestick + volume + MA(5,20,60)")
candlestick(df, volume=True, ma=[5, 20, 60], title="AAPL").save("/tmp/efin_01_basic.png")

# ---------------------------------------------------------------------------
# Example 2: BBands + SAR + Keltner
# ---------------------------------------------------------------------------
print("[2/4] Channels: BBands + SAR + Keltner")
(
    candlestick(df, volume=True, ma=[20], title="AAPL — BBands + SAR + Keltner")
    .add_bbands(20, 2)
    .add_sar()
    .add_keltner(20, 10, 2.0)
    .save("/tmp/efin_02_channels.png")
)

# ---------------------------------------------------------------------------
# Example 3: Multi-panel (MACD + RSI + ADX + Stoch)
# ---------------------------------------------------------------------------
print("[3/4] Multi-panel: MACD + RSI + ADX + Stoch")
(
    candlestick(df.tail(180), volume=True, title="AAPL — MACD + RSI + ADX + Stoch (last 6 months)")
    .add_ma([20, 60])
    .add_macd()
    .add_rsi()
    .add_adx()
    .add_stoch()
    .save("/tmp/efin_03_panels.png")
)

# ---------------------------------------------------------------------------
# Example 4: CSP pattern markers (simulated)
# ---------------------------------------------------------------------------
print("[4/4] CSP pattern markers")
import pandas as pd

n = len(df)
signals_doji = pd.DataFrame({"doji": [False] * n}, index=df.index)
signals_engulf = pd.DataFrame({"engulfing": [False] * n}, index=df.index)
for i in [15, 42, 78, 120, 180, 250, 320]:
    if i < n:
        signals_doji.iloc[i, 0] = True
for i in [30, 65, 100, 150, 200, 280, 350]:
    if i < n:
        signals_engulf.iloc[i, 0] = True

(
    candlestick(df, volume=True, title="AAPL — CSP Patterns (Doji + Engulfing)")
    .add_ma([20, 60])
    .add_pattern(signals_doji, name="Doji", bull=None)
    .add_pattern(signals_engulf, name="Engulfing", bull=None)
    .save("/tmp/efin_04_patterns.png")
)

print("\nCharts saved:")
print("  /tmp/efin_01_basic.png     — Basic candlestick + volume + MA")
print("  /tmp/efin_02_channels.png  — BBands + SAR + Keltner")
print("  /tmp/efin_03_panels.png    — MACD + RSI + ADX + Stoch")
print("  /tmp/efin_04_patterns.png  — CSP pattern markers")
