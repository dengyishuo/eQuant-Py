"""Test add_indicator routing."""
from ebacktestcraft import add_indicator, list_indicators
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=100, freq="B")
codes = ["AAPL", "GOOG", "MSFT"]
rows = []
for dt in dates:
    for code in codes:
        rows.append({
            "date": dt, "code": code, "name": code,
            "open": 100 + np.random.randn(),
            "high": 102 + np.random.randn(),
            "low": 98 + np.random.randn(),
            "close": 100 + np.random.randn(),
            "volume": 1000000 + np.random.randn() * 100000,
            "adjusted": 100 + np.random.randn(),
        })
df = pd.DataFrame(rows)
df["returns"] = df.groupby("code")["close"].pct_change()

# 1. eClassic
r1 = add_indicator(df, "momentum", n=20)
assert "MOM_20" in r1.columns, f"Missing MOM_20: {r1.columns.tolist()}"
print("OK: eClassic.momentum -> MOM_20")

# 2. eTTR
r2 = add_indicator(r1, "rsi", n=14)
assert "RSI_14" in r2.columns, f"Missing RSI_14: {r2.columns.tolist()}"
print("OK: ettr.rsi -> RSI_14")

# 3. eAlpha101
r3 = add_indicator(r2, "alpha001")
assert any("alpha001" in c for c in r3.columns), f"Missing alpha001: {r3.columns.tolist()}"
print("OK: ealpha101.add_alpha001")

# 4. eCandleSticks
r4 = add_indicator(r3, "doji")
assert any("doji" in c.lower() for c in r4.columns), f"Missing doji: {r4.columns.tolist()}"
print("OK: ecandlesticks.add_doji")

# 5. Disambiguation
r5 = add_indicator(df, "eClassic.sma", n=20)
assert "SMA_20" in r5.columns, f"Missing eClassic sma: {r5.columns.tolist()}"
print("OK: eClassic.sma (disambiguation)")

r6 = add_indicator(df, "eTTR.sma", n=20)
assert "SMA_20" in r6.columns, f"Missing eTTR sma: {r6.columns.tolist()}"
print("OK: eTTR.sma (disambiguation)")

r7 = add_indicator(df, "eClassic.volatility", n=20)
assert any("vol" in c.lower() for c in r7.columns), f"Missing vol: {r7.columns.tolist()}"
print("OK: eClassic.volatility (disambiguation)")

# 6. Error handling
try:
    add_indicator(df, "not_exist")
    assert False, "Should raise"
except ValueError as e:
    print(f"OK: Unknown indicator raises ValueError")

# 7. list_indicators
lst = list_indicators()
print(f"OK: list_indicators() -> {len(lst)} indicators")

print("\nAll tests passed!")
