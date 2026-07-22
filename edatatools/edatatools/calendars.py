"""Chinese trading calendar operations.

Port of edatatools R package's date_* functions.

Core concept:
    A TradingCalendar holds a sorted list of business dates for a region
    (e.g., CN = China A-shares). All date operations use nearest-neighbor
    interpolation to map arbitrary dates to the nearest trading day.
"""

import bisect
from datetime import date
from typing import Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Built-in calendar data (fallback when no external source is available)
# China A-share trading days from Tushare trade_cal for 2010-01-01 to 2025-12-31
# ---------------------------------------------------------------------------

def _load_builtin_cn_calendar() -> list:
    """Load China A-share trading calendar, trying akshare first, then Tushare.

    If neither is available, raises ImportError with instructions.
    """
    # Try akshare first (no token needed)
    try:
        import akshare as ak

        df = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(df["trade_date"]).tolist()
        return sorted(dates)
    except Exception:
        pass

    # Try Tushare
    try:
        import tushare as ts

        pro = ts.pro_api()
        df = pro.trade_cal(
            exchange="SSE",
            start_date="20100101",
            end_date="20301231",
            is_open="1",
        )
        dates = pd.to_datetime(df["cal_date"], format="%Y%m%d").tolist()
        return sorted(dates)
    except Exception:
        pass

    raise ImportError(
        "Cannot load Chinese trading calendar. Install akshare (pip install akshare) "
        "or provide a Tushare token. Alternatively, pass dates manually: "
        "TradingCalendar('CN', dates=[...])."
    )


class TradingCalendar:
    """A sorted list of trading days for a given market region.

    Parameters
    ----------
    region : str
        Market region code. e.g., "CN" for China A-shares, "HK" for Hong Kong.
    dates : list of date or pd.DatetimeIndex
        Pre-loaded trading day dates. If None, uses built-in fallback.
    """

    def __init__(self, region: str, dates: Optional[list] = None):
        self.region = region.upper()
        self._dates: list[date] = []
        self._loaded = False
        if dates is not None:
            result = []
            for d in dates:
                if isinstance(d, pd.Timestamp):
                    result.append(d.date())
                elif isinstance(d, date):
                    result.append(d)
                elif isinstance(d, str):
                    result.append(pd.to_datetime(d).date())
                else:
                    result.append(d)
            self._dates = sorted(result)
            self._loaded = True

    def _ensure_loaded(self):
        """Lazy-load calendar data if not yet loaded."""
        if not self._loaded:
            raw = _load_builtin_cn_calendar()
            # Normalize all dates to datetime.date
            result = []
            for d in raw:
                if isinstance(d, pd.Timestamp):
                    result.append(d.date())
                elif isinstance(d, date):
                    result.append(d)
                elif isinstance(d, str):
                    result.append(pd.to_datetime(d).date())
                else:
                    result.append(d)
            self._dates = sorted(result)
            self._loaded = True

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._dates)

    def __contains__(self, d: date) -> bool:
        """Check if date is a trading day."""
        self._ensure_loaded()
        d_std = _to_date(d)
        i = bisect.bisect_left(self._dates, d_std)
        return i < len(self._dates) and self._dates[i] == d_std

    def __getitem__(self, idx: int) -> date:
        self._ensure_loaded()
        return self._dates[idx]

    # ---- Core operations ----

    def to_bus_date(
        self,
        d: Union[date, str, pd.Timestamp],
        shift: int = 0,
        forward: bool = True,
    ) -> Optional[date]:
        """Map any date to the nearest trading day via nearest-neighbor interpolation.

        Equivalent to R's date_to_bus_date(date, region, shift, forward).

        Parameters
        ----------
        d : date-like
            Input date.
        shift : int
            Offset in trading days. 0 = nearest, 1 = next, -1 = previous.
        forward : bool
            If True, prefer the later date when equidistant.

        Returns
        -------
        date or None
        """
        d_std = _to_date(d)
        self._ensure_loaded()

        # Check exact match first — if the date is already a trading day,
        # return it directly (then apply shift).
        i = bisect.bisect_left(self._dates, d_std)
        if i < len(self._dates) and self._dates[i] == d_std:
            pos = i
        else:
            # Nearest-neighbor interpolation (R's approx with method="constant")
            numeric_dates = np.array([_date2num(x) for x in self._dates], dtype=float)
            target = _date2num(d_std)
            idx = np.searchsorted(numeric_dates, target, side="left")
            if idx == 0:
                pos = 0
            elif idx >= len(numeric_dates):
                pos = len(numeric_dates) - 1
            else:
                left = numeric_dates[idx - 1]
                right = numeric_dates[idx]
                if forward:
                    pos = idx  # prefer later
                else:
                    # prefer earlier when equidistant with forward=False
                    if target - left <= right - target:
                        pos = idx - 1
                    else:
                        pos = idx

        pos += shift
        if pos < 0 or pos >= len(self._dates):
            return None
        return self._dates[pos]

    def is_bus_date(self, d: Union[date, str, pd.Timestamp]) -> bool:
        """Check if a date is a trading day.

        Equivalent to R's date_is_bus_date(date, region).
        """
        d_std = _to_date(d)
        return d_std == self.to_bus_date(d_std, shift=0)

    def bus_diff(
        self,
        from_date: Union[date, str, pd.Timestamp],
        to_date: Union[date, str, pd.Timestamp],
    ) -> Optional[int]:
        """Number of trading days between two trading days.

        Equivalent to R's date_bus_diff(fromDate, toDate, region).
        Both dates must be valid trading days.

        Returns
        -------
        int or None if either date is not a trading day.
        """
        fd = _to_date(from_date)
        td = _to_date(to_date)
        self._ensure_loaded()

        idx_from = bisect.bisect_left(self._dates, fd)
        idx_to = bisect.bisect_left(self._dates, td)

        if idx_from >= len(self._dates) or self._dates[idx_from] != fd:
            return None
        if idx_to >= len(self._dates) or self._dates[idx_to] != td:
            return None

        return idx_to - idx_from

    def date_range(
        self,
        start_date: Union[date, str, pd.Timestamp],
        end_date: Optional[Union[date, str, pd.Timestamp]] = None,
        n: Optional[int] = None,
    ) -> list[date]:
        """Generate a range of business dates.

        Equivalent to R's date_range(startDate, endDate, n, region).

        Must provide exactly one of end_date or n.

        Parameters
        ----------
        start_date : date-like
            Start date (will be mapped to nearest trading day).
        end_date : date-like, optional
            End date (inclusive if it is a trading day).
        n : int, optional
            Number of trading days to return.

        Returns
        -------
        list of date
        """
        if (end_date is None) == (n is None):
            raise ValueError("Must provide exactly one of end_date or n")

        start_date = _to_date(start_date)
        sd = self.to_bus_date(start_date, shift=0)
        if sd is None:
            return []

        start_idx = bisect.bisect_left(self._dates, sd)
        if end_date is not None:
            ed = _to_date(end_date)
            if not self.is_bus_date(ed):
                ed = self.to_bus_date(ed, shift=-1)
            if ed is None or ed < sd:
                return []
            end_idx = bisect.bisect_left(self._dates, ed)
            return self._dates[start_idx: end_idx + 1]
        else:
            if n <= 0:
                raise ValueError("n must be positive")
            return self._dates[start_idx: start_idx + n]

    def shift_months(self, months: int, shift: int) -> int:
        """Shift a YYYYMM integer by a number of months.

        Equivalent to R's date_shift_months(months, shift).

        Parameters
        ----------
        months : int
            Integer in YYYYMM format (e.g., 202501).
        shift : int
            Number of months to shift (can be negative).

        Returns
        -------
        int in YYYYMM format.
        """
        y = months // 100
        m = months % 100
        julian = y * 12 + m
        julian += shift
        new_y = (julian - 1) // 12
        new_m = (julian - 1) % 12 + 1
        return new_y * 100 + new_m

    @classmethod
    def from_tushare(cls, region: str = "CN", token: str = None, **kwargs):
        """Create a TradingCalendar from Tushare's trade_cal endpoint.

        Parameters
        ----------
        region : str
            "CN" for China A-shares (exchange='SSE').
        token : str
            Tushare Pro token.
        **kwargs
            Extra arguments passed to tushare.pro_api().
        """
        import tushare as ts

        pro = ts.pro_api(token) if token else ts.pro_api(**kwargs)
        df = pro.trade_cal(
            exchange="SSE" if region.upper() == "CN" else "",
            start_date="20100101",
            end_date="20301231",
            is_open="1",
        )
        dates = sorted(pd.to_datetime(df["cal_date"], format="%Y%m%d").tolist())
        return cls(region=region, dates=dates)

    def first_bus_date_of_month(self, d: Union[date, str, pd.Timestamp]) -> Optional[date]:
        """First trading day of a given month."""
        d_std = _to_date(d)
        ym = d_std.replace(day=1)
        return self.to_bus_date(ym, shift=0)

    def last_bus_date_of_month(self, d: Union[date, str, pd.Timestamp]) -> Optional[date]:
        """Last trading day of a given month."""
        d_std = _to_date(d)
        # last day of month
        if d_std.month == 12:
            ym = date(d_std.year + 1, 1, 1)
        else:
            ym = date(d_std.year, d_std.month + 1, 1)
        # one day before first of next month
        last_day = ym - pd.Timedelta(days=1)
        return self.to_bus_date(last_day.date(), shift=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_date(d: Union[date, str, pd.Timestamp, int]) -> date:
    """Convert any date-like input to a datetime.date."""
    # NOTE: pd.Timestamp is a subclass of datetime.date, so check it FIRST.
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, (int, np.integer)):
        s = str(int(d))
        return pd.to_datetime(s, format="%Y%m%d").date()
    if isinstance(d, str):
        d_clean = d.replace("-", "").replace("/", "")[:8]
        return pd.to_datetime(d_clean, format="%Y%m%d").date()
    raise TypeError(f"Cannot convert {type(d)} to date")


def _date2num(d: date) -> float:
    """Convert date to numeric (ordinal) for interpolation."""
    return d.toordinal()


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

cn_calendar = TradingCalendar("CN")

# ---------------------------------------------------------------------------
# Module-level convenience functions (matching R edatatools API)
# ---------------------------------------------------------------------------

def date_to_bus_date(
    d: Union[date, str, pd.Timestamp],
    region: str = "CN",
    shift: int = 0,
    forward: bool = True,
) -> Optional[date]:
    """Convert any date to the nearest trading day.

    Equivalent to R's date_to_bus_date(date, region, shift, forward).
    """
    cal = _get_calendar(region)
    return cal.to_bus_date(d, shift=shift, forward=forward)


def date_is_bus_date(
    d: Union[date, str, pd.Timestamp],
    region: str = "CN",
) -> bool:
    """Check if a date is a trading day.

    Equivalent to R's date_is_bus_date(date, region).
    """
    cal = _get_calendar(region)
    return cal.is_bus_date(d)


def date_bus_diff(
    from_date: Union[date, str, pd.Timestamp],
    to_date: Union[date, str, pd.Timestamp],
    region: str = "CN",
) -> Optional[int]:
    """Number of trading days between two trading days.

    Equivalent to R's date_bus_diff(fromDate, toDate, region).
    """
    cal = _get_calendar(region)
    return cal.bus_diff(from_date, to_date)


def date_range(
    start_date: Union[date, str, pd.Timestamp],
    end_date: Optional[Union[date, str, pd.Timestamp]] = None,
    n: Optional[int] = None,
    region: str = "CN",
) -> list[date]:
    """Generate a range of business dates.

    Equivalent to R's date_range(startDate, endDate, n, region).
    """
    cal = _get_calendar(region)
    return cal.date_range(start_date, end_date=end_date, n=n)


# ---------------------------------------------------------------------------
# Internal registry
# ---------------------------------------------------------------------------

_calendars: dict[str, TradingCalendar] = {
    "CN": cn_calendar,
}


def _get_calendar(region: str) -> TradingCalendar:
    region = region.upper()
    if region not in _calendars:
        raise ValueError(
            f"No calendar registered for region '{region}'. "
            f"Register one via register_calendar(region, calendar)."
        )
    return _calendars[region]


def register_calendar(region: str, calendar: TradingCalendar):
    """Register a TradingCalendar for a region."""
    _calendars[region.upper()] = calendar
