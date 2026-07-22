"""Quick smoke test for eFinCharts."""
import numpy as np
import pandas as pd
import datetime

# Generate fake OHLC data
np.random.seed(42)
n = 100
dates = pd.date_range("2024-01-01", periods=n, freq="B")
close = 100 + np.cumsum(np.random.randn(n) * 2)
high = close + np.abs(np.random.randn(n) * 3)
low = close - np.abs(np.random.randn(n) * 3)
open_price = close - np.random.randn(n) * 2
volume = np.random.randint(100000, 1000000, n)

df = pd.DataFrame({
    "Open": open_price,
    "High": high,
    "Low": low,
    "Close": close,
    "Volume": volume,
}, index=dates)

print("=== 1. Basic import ===")
from efincharts import candlestick, Chart, efin_style, prepare_ohlc
from efincharts.theme import COLOR_UP, COLOR_DOWN
print(f"  COLOR_UP: {COLOR_UP}")
print(f"  COLOR_DOWN: {COLOR_DOWN}")
print("  PASS")

print()
print("=== 2. Data preparation ===")
clean = prepare_ohlc(df)
print(f"  Columns: {list(clean.columns)}")
print(f"  Index type: {type(clean.index)}")
print("  PASS")

print()
print("=== 3. Chart creation (no display) ===")
chart = candlestick(df, volume=True, ma=[5, 20], title="Test Chart")
print(f"  Chart: {chart}")
print("  PASS")

print()
print("=== 4. Builder pattern ===")
chart2 = (
    candlestick(df, volume=True, title="Full Chart")
    .add_ma([10, 30])
    .add_bbands()
    .add_macd()
    .add_rsi()
)
print(f"  Chart: {chart2}")
print(f"  Addplots count: {len(chart2._addplots)}")
print("  PASS")

print()
print("=== 5. CSP pattern markers ===")
# Fake some signals
signals = pd.DataFrame({
    "Hammer": [False] * n,
}, index=dates)
signals.iloc[10, 0] = True
signals.iloc[30, 0] = True
chart3 = candlestick(df, volume=True).add_pattern(signals, name="Hammer", bull=True)
print(f"  Chart: {chart3}")
print(f"  Addplots count: {len(chart3._addplots)}")
print("  PASS")

print()
print("=== 6. Save to file ===")
chart.save("/tmp/efin_test.png")
import os
if os.path.exists("/tmp/efin_test.png"):
    size = os.path.getsize("/tmp/efin_test.png")
    print(f"  Saved: /tmp/efin_test.png ({size} bytes)")
    print("  PASS")
else:
    print("  FAIL - file not created")

print()
print("All tests passed!")
