"""Shared test fixtures for eFactorCraft."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def mock_panel() -> pd.DataFrame:
    """Small panel: 3 assets x 60 trading days with OHLCV + fundamental data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    codes = ["AAPL", "MSFT", "GOOG"]
    names = ["Apple", "Microsoft", "Alphabet"]

    rows = []
    for d in dates:
        base_p = np.random.uniform(100, 200, 3)
        for i, (c, n) in enumerate(zip(codes, names)):
            rows.append(
                {
                    "date": d,
                    "code": c,
                    "name": n,
                    "open": base_p[i] * np.random.uniform(0.98, 1.02),
                    "high": base_p[i] * np.random.uniform(1.01, 1.05),
                    "low": base_p[i] * np.random.uniform(0.95, 0.99),
                    "close": base_p[i] * np.random.uniform(0.99, 1.01),
                    "adjusted": base_p[i] * np.random.uniform(0.99, 1.01),
                    "volume": np.random.uniform(1e6, 1e7),
                    "cap": np.random.uniform(1e10, 1e13),
                    "bv": np.random.uniform(1e9, 1e11),
                }
            )
    return pd.DataFrame(rows)
