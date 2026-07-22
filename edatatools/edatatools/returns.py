"""CACS cumulative return calculation.

Port of edatatools R package's calCumRet function.

The formula correctly handles China A-share adjustment factors (ex-rights/ex-dividends):

    cum_ret = (close.e * factor.e + (const.e - const.s)) / factor.s / close.s - 1

where:
    close.s = close price at start date
    close.e = close price at end date
    factor.s = cumulative adjustment factor before start date
    factor.e = cumulative adjustment factor before end date
    const.s = cumulative adjustment constant before start date
    const.e = cumulative adjustment constant before end date
"""

from datetime import date
from typing import Optional, Union

import numpy as np
import pandas as pd


def cal_cum_ret(
    jtids: list[str],
    start_dates: Union[str, date, pd.Timestamp, list],
    end_dates: Union[str, date, pd.Timestamp, list],
    close: pd.DataFrame,
    adj_factor: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Calculate CACS-adjusted cumulative returns.

    Parameters
    ----------
    jtids : list of str
        Stock symbols (e.g., JTID codes).
    start_dates : date-like or list of date-like
        Start date(s). If single value, broadcast to all jtids.
    end_dates : date-like or list of date-like
        End date(s). If single value, broadcast to all jtids.
    close : pd.DataFrame
        Close price data. Must have index=date, columns=jtid.
    adj_factor : pd.DataFrame, optional
        Adjustment factor data with columns:
        ['jtid', 'exDiviDate', 'adjustingFactor', 'adjustingConst'].
        If None, returns simple return: close.e / close.s - 1.

    Returns
    -------
    np.ndarray of shape (len(jtids),)
        CACS-adjusted cumulative returns. NaN where data is missing.

    Examples
    --------
    >>> from edatatools import cal_cum_ret
    >>> rets = cal_cum_ret(
    ...     jtids=['000001', '000002'],
    ...     start_dates='20240101',
    ...     end_dates='20241231',
    ...     close=close_df,
    ...     adj_factor=adj_df,
    ... )
    """
    n = len(jtids)

    # Normalize inputs
    if isinstance(start_dates, (str, date, pd.Timestamp)):
        start_dates = [start_dates] * n
    if isinstance(end_dates, (str, date, pd.Timestamp)):
        end_dates = [end_dates] * n

    start_dates = pd.DatetimeIndex(start_dates)
    end_dates = pd.DatetimeIndex(end_dates)

    if len(start_dates) != n or len(end_dates) != n:
        raise ValueError("Length of start_dates/end_dates must be 1 or equal to len(jtids)")

    if (start_dates > end_dates).any():
        raise ValueError("All start_dates must be <= end_dates")

    # Initialize result
    result = np.full(n, np.nan)

    # Default adjustment values
    factor_s = np.ones(n)
    factor_e = np.ones(n)
    const_s = np.zeros(n)
    const_e = np.zeros(n)

    # Extract close prices
    close_s = np.full(n, np.nan)
    close_e = np.full(n, np.nan)

    for i, jtid in enumerate(jtids):
        if jtid in close.columns:
            sd, ed = start_dates[i], end_dates[i]
            row_s = close.index.asof(sd)
            row_e = close.index.asof(ed)
            if row_s is not None:
                close_s[i] = close.at[row_s, jtid]
            if row_e is not None:
                close_e[i] = close.at[row_e, jtid]

    # Apply adjustment factors if provided
    if adj_factor is not None:
        for i, jtid in enumerate(jtids):
            sd = start_dates[i]
            ed = end_dates[i]

            adj_jtid = adj_factor[adj_factor['jtid'] == jtid]

            # Factors up to start date
            adj_s = adj_jtid[adj_jtid['exDiviDate'] <= sd]
            if len(adj_s) > 0:
                last = adj_s.iloc[-1]
                factor_s[i] = last['adjustingFactor']
                const_s[i] = last['adjustingConst']

            # Factors up to end date
            adj_e = adj_jtid[adj_jtid['exDiviDate'] <= ed]
            if len(adj_e) > 0:
                last = adj_e.iloc[-1]
                factor_e[i] = last['adjustingFactor']
                const_e[i] = last['adjustingConst']

    # CACS formula: (close.e * factor.e + (const.e - const.s)) / factor.s / close.s - 1
    mask = ~np.isnan(close_s) & ~np.isnan(close_e) & (factor_s != 0) & (close_s != 0)
    result[mask] = (
        (close_e[mask] * factor_e[mask] + (const_e[mask] - const_s[mask]))
        / factor_s[mask]
        / close_s[mask]
        - 1.0
    )

    return result
