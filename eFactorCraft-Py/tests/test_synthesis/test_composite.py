"""Tests for factor synthesis composite functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from efactorcraft import synthesis
import eclassic as factor
import efactorcraft as eng


@pytest.fixture(scope="module")
def panel_with_factors(mock_panel):
    """Panel with multiple factors and forward returns (3 stocks, basic tests)."""
    df = factor.momentum(mock_panel, n=[5, 10])
    df = factor.value(df, bv_col="bv", cap_col="cap")
    df = factor.size(df, cap_col="cap")
    df = factor.volatility(df, close_col="adjusted", n=10)
    df = eng.add_next_return(df, close_col="adjusted", periods=[1, 5])
    return df


@pytest.fixture(scope="module")
def large_panel():
    """Larger panel (6 stocks x 100 days) for IC-based tests."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    codes = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA"]
    rows = []
    for d in dates:
        for c in codes:
            bp = np.random.uniform(100, 400)
            rows.append({
                "date": d, "code": c, "name": c,
                "open": bp * np.random.uniform(0.98, 1.02),
                "high": bp * np.random.uniform(1.01, 1.05),
                "low": bp * np.random.uniform(0.95, 0.99),
                "close": bp * np.random.uniform(0.99, 1.01),
                "adjusted": bp * np.random.uniform(0.99, 1.01),
                "volume": np.random.uniform(1e6, 1e7),
                "cap": np.random.uniform(1e10, 1e12),
                "bv": np.random.uniform(1e9, 1e11),
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    df = factor.momentum(df, n=[5, 10])
    df = factor.value(df, bv_col="bv", cap_col="cap")
    df = factor.size(df, cap_col="cap")
    df = factor.volatility(df, close_col="adjusted", n=10)
    df = eng.add_next_return(df, close_col="adjusted", periods=[1, 5])
    return df


class TestEqualWeighted:
    def test_basic(self, panel_with_factors):
        result = synthesis.equal_weighted_composite(
            panel_with_factors, factor_cols=["mom_5", "mom_10", "value"]
        )
        assert "composite_ew" in result.columns
        assert result["composite_ew"].notna().any()

    def test_append_false(self, panel_with_factors):
        result = synthesis.equal_weighted_composite(
            panel_with_factors, factor_cols=["mom_5", "value"], append=False
        )
        assert set(result.columns) == {"date", "code", "name", "composite_ew"}

    def test_single_factor(self, panel_with_factors):
        result = synthesis.equal_weighted_composite(
            panel_with_factors, factor_cols=["mom_5"]
        )
        assert result["composite_ew"].notna().any()


class TestRankWeighted:
    def test_basic(self, panel_with_factors):
        result = synthesis.rank_weighted_composite(
            panel_with_factors, factor_cols=["mom_5", "mom_10", "size"]
        )
        assert "composite_rw" in result.columns
        vals = result["composite_rw"].dropna()
        assert 0 <= vals.min() <= 1
        assert 0 <= vals.max() <= 1

    def test_outlier_robust(self, panel_with_factors):
        df = panel_with_factors.copy()
        df.loc[0, "mom_5"] = 1e6
        result = synthesis.rank_weighted_composite(
            df, factor_cols=["mom_5", "mom_10"]
        )
        vals = result["composite_rw"].dropna()
        assert 0 <= vals.max() <= 1


class TestICWeighted:
    def test_basic(self, large_panel):
        result = synthesis.ic_weighted_composite(
            large_panel, factor_cols=["mom_5", "mom_10", "value"],
            forward_col="forward_5", window=30
        )
        assert "composite_icw" in result.columns
        assert result["composite_icw"].notna().any()

    def test_auto_detect_forward(self, large_panel):
        result = synthesis.ic_weighted_composite(
            large_panel, factor_cols=["mom_5", "value"], window=30
        )
        assert "composite_icw" in result.columns

    def test_no_forward_raises(self, mock_panel):
        with pytest.raises(ValueError, match="forward"):
            synthesis.ic_weighted_composite(mock_panel, factor_cols=["close"])


class TestICIRWeighted:
    def test_basic(self, large_panel):
        result = synthesis.icir_weighted_composite(
            large_panel, factor_cols=["mom_5", "mom_10"],
            forward_col="forward_5", window=30
        )
        assert "composite_icir" in result.columns
        assert result["composite_icir"].notna().any()

    def test_small_window(self, large_panel):
        result = synthesis.icir_weighted_composite(
            large_panel, factor_cols=["mom_5"],
            forward_col="forward_5", window=15
        )
        assert "composite_icir" in result.columns


class TestPCA:
    def test_basic(self, large_panel):
        result = synthesis.pca_composite(
            large_panel, factor_cols=["mom_5", "mom_10", "value", "size"],
            min_periods=5
        )
        assert "composite_pca" in result.columns
        assert result["composite_pca"].notna().any()

    def test_too_few_assets(self, panel_with_factors):
        df = panel_with_factors.head(20).copy()
        result = synthesis.pca_composite(
            df, factor_cols=["mom_5", "mom_10", "value"],
            min_periods=5
        )
        assert "composite_pca" in result.columns


class TestMaxDecay:
    def test_basic(self, large_panel):
        result = synthesis.max_decay_composite(
            large_panel, factor_cols=["mom_5", "mom_10", "value"],
            forward_col="forward_5", window=30
        )
        assert "composite_md" in result.columns
        assert result["composite_md"].notna().any()

    def test_all_negative_ic(self, large_panel):
        result = synthesis.max_decay_composite(
            large_panel, factor_cols=["mom_5"],
            forward_col="forward_5", window=10
        )
        assert "composite_md" in result.columns
