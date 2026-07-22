"""Smoke test for all edatatools modules."""
from edatatools import (
    TradingCalendar,
    cal_cum_ret,
    build_universe,
    register_calendar,
)
import pandas as pd
import numpy as np
import datetime

# 1. cal_cum_ret
print("=== cal_cum_ret ===")
close_df = pd.DataFrame(
    {"000001": [10.0, 11.0, 12.0, 13.0], "000002": [20.0, 21.0, 19.0, 22.0]},
    index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
)
ret = cal_cum_ret(["000001", "000002"], "2024-01-02", "2024-01-05", close_df)
print(f"  returns: {ret} (expected [0.3, 0.1])")
assert abs(ret[0] - 0.3) < 1e-9
assert abs(ret[1] - 0.1) < 1e-9
print("  PASS")

# 2. build_universe (fixed)
print()
print("=== build_universe (fixed) ===")
dates = [
    datetime.date(2024, 1, d)
    for d in range(1, 32)
    if datetime.date(2024, 1, d).weekday() < 5
]
cal = TradingCalendar("CN", dates=dates)
print(f"  cal._dates type: {type(cal._dates[0])}")
print(f"  cal._loaded: {cal._loaded}")
register_calendar("CN", cal)

# Verify registry
from edatatools.calendars import _calendars, _get_calendar
cal2 = _get_calendar("CN")
print(f"  registry cal._dates type: {type(cal2._dates[0])}")
print(f"  same object: {cal is cal2}")

print(f"  start_date=pd.Timestamp('2024-01-05') type: {type(pd.Timestamp('2024-01-05'))}")
print("  calling build_universe...")
univ = build_universe(
    "2024-01-05", "2024-01-10",
    univ_type="fixed",
    fixed_list=["000001", "000002", "000003"],
)
print(f"  dates count: {len(univ)}")
print(f"  20240108 stocks: {univ['20240108']}")
assert univ["20240108"] == ["000001", "000002", "000003"]
print("  PASS")

print()
print("All module tests passed!")
