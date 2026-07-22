"""edatatools — Data infrastructure for Chinese quantitative research.

Core modules:
- calendars: Chinese trading calendar operations
- returns: CACS cumulative return calculation
- universe: Stock universe / pool construction
"""

from .calendars import (
    TradingCalendar,
    cn_calendar,
    date_bus_diff,
    date_is_bus_date,
    date_range,
    date_to_bus_date,
    register_calendar,
)

from .returns import cal_cum_ret

from .universe import (
    build_universe,
)

__all__ = [
    # Calendars
    "TradingCalendar",
    "cn_calendar",
    "date_to_bus_date",
    "date_is_bus_date",
    "date_bus_diff",
    "date_range",
    "register_calendar",
    # Returns
    "cal_cum_ret",
    # Universe
    "build_universe",
]
