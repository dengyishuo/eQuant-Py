"""Risk-Adjusted Momentum — eClassic add_ram equivalent."""

from __future__ import annotations

from typing import Optional, Literal

import numpy as np
import pandas as pd

from eclassic._panel import _resolve_col
from equant.utils.panel import slim_output, validate_panel


def ram(
    df,
    close_col: Optional[str] = None,
    n: int = 252,
    risk: Literal["vol", "VaR", "CVaR"] = "vol",
    new_col: str = "ram",
    append: bool = True,
):
    """Risk-Adjusted Momentum.

    ``RAM = n-period return / risk_measure``

    Parameters
    ----------
    n : int
        Lookback period.
    risk : str
        ``"vol"`` = std dev, ``"VaR"`` = 5th percentile, ``"CVaR"`` = expected shortfall.
    """
    validate_panel(df)
    col = _resolve_col(df, "close", close_col)
    cname = f"{new_col}_{n}"
    result = df.copy()
    result[cname] = np.nan

    for _code, idx in result.groupby("code", sort=False).groups.items():
        sub = result.loc[idx].sort_values("date")
        vals = sub[col].values.astype(np.float64)

        rets = np.full(len(vals), np.nan)
        rets[n:] = (vals[n:] - vals[:-n]) / np.maximum(np.abs(vals[:-n]), 1e-15)

        rolled = pd.Series(rets).rolling(window=n, min_periods=n)

        if risk == "vol":
            risk_measure = rolled.std(ddof=1).values
        elif risk == "VaR":
            risk_measure = rolled.quantile(0.05).values
        elif risk == "CVaR":
            def _cvar(w):
                w = w.dropna()
                if len(w) < 2:
                    return np.nan
                thresh = w.quantile(0.05)
                return w[w <= thresh].mean()
            risk_measure = rolled.apply(_cvar, raw=False).values
        else:
            raise ValueError(f"Unknown risk measure: {risk}")

        result.loc[sub.index, cname] = np.where(
            np.abs(risk_measure) > 1e-10,
            rets / risk_measure,
            np.nan,
        )

    return slim_output(result, cname, append)
