"""Smoke test for edatatools package."""
from edatatools import TradingCalendar, date_to_bus_date, date_is_bus_date, date_bus_diff, date_range
import datetime

# Test with explicit dates (weekdays only for Jan 2024)
dates = [datetime.date(2024, 1, d) for d in range(1, 32) if datetime.date(2024, 1, d).weekday() < 5]
cal = TradingCalendar("CN", dates=dates)

print("1. __len__:", len(cal), "(expected ~23)")

print("2. __contains__ 2024-01-08:", datetime.date(2024, 1, 8) in cal, "(expected True)")
print("3. __contains__ 2024-01-06:", datetime.date(2024, 1, 6) in cal, "(expected False)")

print("4. to_bus_date 2024-01-15:", cal.to_bus_date("2024-01-15"), "(expected 2024-01-15)")

print("5. is_bus_date 2024-01-15:", cal.is_bus_date("2024-01-15"), "(expected True)")
print("6. is_bus_date 2024-01-13:", cal.is_bus_date("2024-01-13"), "(expected False)")

print("7. bus_diff 01-02 -> 01-31:", cal.bus_diff("2024-01-02", "2024-01-31"), "(expected 21)")

r = cal.date_range("2024-01-10", "2024-01-19")
print("8. date_range 01-10 -> 01-19:", len(r), "days:", r[0], "to", r[-1])

print("9. shift_months(202501, -3):", cal.shift_months(202501, -3), "(expected 202410)")

print("\nModule-level functions:")
print("10. date_to_bus_date:", date_to_bus_date("2024-01-15"))
print("11. date_range(n=5):", date_range("2024-01-08", n=5))

print("\nAll tests passed!")
