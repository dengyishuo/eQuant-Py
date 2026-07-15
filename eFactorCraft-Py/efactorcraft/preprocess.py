"""Factor preprocessing pipeline — eFactorCraft equivalent.

Winsorization, standardization, industry/size neutralization,
and the one-stop ``factor_preprocess()`` pipeline.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from equant.utils.panel import slim_output, sort_panel, validate_panel


def winsorize(
    df: pd.DataFrame,
    factor_col: Union[str, Sequence[str]],
    probs: tuple[float, float] = (0.01, 0.99),
    by: str = "date",
    new_col_prefix: str = "win",
    append: bool = True,
) -> pd.DataFrame:
    """Cross-sectional winsorization — clip to quantile bounds.

    Parameters
    ----------
    factor_col : str or sequence
        Factor column(s) to winsorize.
    probs : tuple
        Lower and upper quantile bounds.
    by : str
        Grouping column (usually ``"date"``).
    new_col_prefix : str
        Prefix for output columns (e.g., ``"win_value"``).
    """
    validate_panel(df)
    cols = [factor_col] if isinstance(factor_col, str) else list(factor_col)
    result = df.copy()

    for col in cols:
        out_name = f"{new_col_prefix}_{col}"

        def _winsor(g):
            lo = g[col].quantile(probs[0])
            hi = g[col].quantile(probs[1])
            return g[col].clip(lo, hi)

        result[out_name] = result.groupby(by, group_keys=False).apply(_winsor, include_groups=False)

    out_cols = [f"{new_col_prefix}_{c}" for c in cols]
    return slim_output(result, out_cols, append)


def standardize(
    df: pd.DataFrame,
    factor_col: Union[str, Sequence[str]],
    by: str = "date",
    new_col_prefix: str = "std",
    append: bool = True,
) -> pd.DataFrame:
    """Cross-sectional z-score standardization.

    ``z = (x - mean(x)) / std(x)``
    """
    validate_panel(df)
    cols = [factor_col] if isinstance(factor_col, str) else list(factor_col)
    result = df.copy()

    for col in cols:
        out_name = f"{new_col_prefix}_{col}"
        g = result.groupby(by)[col]
        mean = g.transform("mean")
        std = g.transform("std")
        result[out_name] = np.where(std > 1e-15, (result[col] - mean) / std, np.nan)

    out_cols = [f"{new_col_prefix}_{c}" for c in cols]
    return slim_output(result, out_cols, append)


def industry_neutralize(
    df: pd.DataFrame,
    factor_col: Union[str, Sequence[str]],
    industry_col: str = "industry",
    by: str = "date",
    new_col_prefix: str = "ind_neu",
    min_samples: int = 5,
    append: bool = True,
) -> pd.DataFrame:
    """Industry neutralization via cross-sectional OLS per date.

    ``residual = y - X * beta`` where X = industry dummies.
    """
    import statsmodels.api as sm

    validate_panel(df)
    cols = [factor_col] if isinstance(factor_col, str) else list(factor_col)
    result = df.copy()

    for col in cols:
        out_name = f"{new_col_prefix}_{col}"
        result[out_name] = np.nan

        for date, group in result.groupby(by):
            y = group[col].values
            valid = ~np.isnan(y)
            if valid.sum() < min_samples:
                continue

            industries = group[industry_col].values
            # Handle NaN industry
            ind_valid = ~pd.isna(industries)
            combined = valid & ind_valid
            if combined.sum() < min_samples:
                continue

            X = pd.get_dummies(industries[combined]).astype(float)
            if X.shape[1] < 2:
                continue

            model = sm.OLS(y[combined], X).fit()
            resid = np.full(len(group), np.nan)
            resid[combined] = model.resid
            result.loc[group.index, out_name] = resid

    out_cols = [f"{new_col_prefix}_{c}" for c in cols]
    return slim_output(result, out_cols, append)


def size_neutralize(
    df: pd.DataFrame,
    factor_col: Union[str, Sequence[str]],
    size_col: str = "cap",
    by: str = "date",
    new_col_prefix: str = "size_neu",
    min_samples: int = 5,
    append: bool = True,
) -> pd.DataFrame:
    """Size (market cap) neutralization via cross-sectional OLS per date.

    ``residual = y - alpha - beta * log(size)``
    """
    import statsmodels.api as sm

    validate_panel(df)
    cols = [factor_col] if isinstance(factor_col, str) else list(factor_col)
    result = df.copy()

    for col in cols:
        out_name = f"{new_col_prefix}_{col}"
        result[out_name] = np.nan

        for date, group in result.groupby(by):
            y = group[col].values
            sz = group[size_col].values
            log_size = np.log(np.maximum(sz, 1e-10))
            valid = ~(np.isnan(y) | np.isnan(log_size))
            if valid.sum() < min_samples:
                continue

            X = sm.add_constant(log_size[valid])
            model = sm.OLS(y[valid], X).fit()
            resid = np.full(len(group), np.nan)
            resid[valid] = model.resid
            result.loc[group.index, out_name] = resid

    out_cols = [f"{new_col_prefix}_{c}" for c in cols]
    return slim_output(result, out_cols, append)


def factor_preprocess(
    df: pd.DataFrame,
    factor_col: str,
    industry_col: Optional[str] = None,
    size_col: Optional[str] = None,
    probs: tuple[float, float] = (0.01, 0.99),
    by: str = "date",
    append: bool = True,
) -> pd.DataFrame:
    """One-stop factor preprocessing pipeline.

    ``winsorize → standardize → industry_neutralize → size_neutralize``

    Steps are only applied if the corresponding column is provided.
    Returns a unique output column named ``full_neu_{factor_col}``.
    """
    validate_panel(df)
    result = df.copy()
    working_col = factor_col

    # Step 1: Winsorize
    result = winsorize(result, working_col, probs=probs, by=by, new_col_prefix="win", append=True)
    working_col = f"win_{factor_col}"

    # Step 2: Standardize
    result = standardize(result, working_col, by=by, new_col_prefix="std", append=True)
    working_col = f"std_win_{factor_col}"

    # Step 3: Industry neutralize (optional)
    if industry_col is not None and industry_col in result.columns:
        result = industry_neutralize(result, working_col, industry_col=industry_col,
                                     by=by, new_col_prefix="ind_neu", append=True)
        working_col = f"ind_neu_{working_col}"

    # Step 4: Size neutralize (optional)
    if size_col is not None and size_col in result.columns:
        result = size_neutralize(result, working_col, size_col=size_col,
                                 by=by, new_col_prefix="size_neu", append=True)
        working_col = f"size_neu_{working_col}"

    # Rename final column
    final_name = f"full_neu_{factor_col}"
    result[final_name] = result[working_col]

    return slim_output(result, final_name, append)
