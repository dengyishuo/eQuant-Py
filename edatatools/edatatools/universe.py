"""Stock universe / pool construction.

Port of edatatools R package's univ function.

Supported universe types:
- liquidityBased: Filter stocks by liquidity score range.
- CSI300: Constituents of CSI 300 index.
- Fixed: Fixed list of stocks.
"""

from datetime import date
from typing import Optional, Union

import pandas as pd


def build_universe(
    start_date: Union[date, str, pd.Timestamp],
    end_date: Optional[Union[date, str, pd.Timestamp]] = None,
    univ_type: str = "liquidity_based",
    univ_range: tuple[float, float] = (0, 1),
    fixed_list: Optional[list[str]] = None,
    liquidity_data: Optional[pd.DataFrame] = None,
    index_constituents: Optional[dict] = None,
    days_smooth: int = 10,
) -> dict[str, list[str]]:
    """Build a stock universe for each trading day in the date range.

    Parameters
    ----------
    start_date : date-like
        Start date.
    end_date : date-like, optional
        End date. Defaults to start_date.
    univ_type : str
        Type of universe:
        - "liquidity_based": Filter by liquidity score percentile.
        - "csi300": CSI 300 index constituents.
        - "fixed": Fixed list of stocks.
    univ_range : tuple of float
        Percentile range for liquidity-based universe. Default (0, 1) = all.
    fixed_list : list of str, optional
        Fixed stock list (required when univ_type='fixed').
    liquidity_data : pd.DataFrame, optional
        Daily liquidity data with columns: [date, jtid, liquidityScore].
        Required when univ_type='liquidity_based'.
    index_constituents : dict of date->list[str], optional
        Pre-loaded index constituents. Required when univ_type='csi300'.
    days_smooth : int
        Number of days to smooth for liquidity-based universe. Default 10.

    Returns
    -------
    dict[str, list[str]]
        Mapping of date (YYYYMMDD) -> list of stock symbols.
    """
    if end_date is None:
        end_date = start_date

    sd = pd.Timestamp(start_date)
    ed = pd.Timestamp(end_date)

    if univ_type == "liquidity_based":
        return _liquidity_universe(
            sd, ed, univ_range, liquidity_data, days_smooth
        )
    elif univ_type.upper() == "CSI300":
        return _csi300_universe(sd, ed, index_constituents)
    elif univ_type == "fixed":
        return _fixed_universe(sd, ed, fixed_list)
    else:
        raise ValueError(f"Unsupported univ_type: {univ_type}")


def _liquidity_universe(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    univ_range: tuple[float, float],
    liquidity_data: Optional[pd.DataFrame],
    days_smooth: int,
) -> dict[str, list[str]]:
    """Build universe by filtering on liquidity score."""
    if liquidity_data is None:
        raise ValueError("liquidity_data is required for liquidity_based universe")

    lb, ub = min(univ_range), max(univ_range)

    # Filter by score range for each day
    daily_univ: dict[str, list[str]] = {}
    for d, grp in liquidity_data.groupby("date"):
        mask = (grp["liquidityScore"] > lb) & (grp["liquidityScore"] <= ub)
        daily_univ[str(d)] = sorted(grp.loc[mask, "jtid"].tolist())

    # Smooth by intersecting over a rolling window
    from .calendars import _get_calendar
    cal = _get_calendar("CN")

    result: dict[str, list[str]] = {}
    all_dates = cal.date_range(start_date, end_date)

    for d in all_dates:
        d_str = d.strftime("%Y%m%d")
        # Gather dates in the smoothing window
        window_dates = cal.date_range(
            cal.to_bus_date(d, shift=-(days_smooth - 1)),
            d,
        )
        # Union over window (take all stocks that appear)
        stocks = set()
        for wd in window_dates:
            wd_str = wd.strftime("%Y%m%d")
            if wd_str in daily_univ:
                stocks.update(daily_univ[wd_str])
        result[d_str] = sorted(stocks)

    return result


def _csi300_universe(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    index_constituents: Optional[dict],
) -> dict[str, list[str]]:
    """Build universe from CSI 300 index constituents."""
    if index_constituents is None:
        raise ValueError("index_constituents is required for CSI300 universe")

    from .calendars import _get_calendar
    cal = _get_calendar("CN")

    result: dict[str, list[str]] = {}
    all_dates = cal.date_range(start_date, end_date)

    for d in all_dates:
        d_str = d.strftime("%Y%m%d")
        # Find the nearest constituent date
        for offset in range(0, -10, -1):
            ref = cal.to_bus_date(d, shift=offset)
            ref_str = ref.strftime("%Y%m%d") if ref else ""
            if ref_str in index_constituents:
                result[d_str] = index_constituents[ref_str]
                break
        else:
            result[d_str] = []

    return result


def _fixed_universe(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    fixed_list: Optional[list[str]],
) -> dict[str, list[str]]:
    """Build universe from a fixed list of stocks."""
    if fixed_list is None:
        raise ValueError("fixed_list is required for fixed universe")

    from .calendars import _get_calendar
    cal = _get_calendar("CN")

    result: dict[str, list[str]] = {}
    all_dates = cal.date_range(start_date, end_date)

    for d in all_dates:
        result[d.strftime("%Y%m%d")] = list(fixed_list)

    return result
