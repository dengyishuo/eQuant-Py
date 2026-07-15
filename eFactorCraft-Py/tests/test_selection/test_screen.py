"""Tests for factor selection screen functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from efactorcraft import selection
import eclassic as factor
import efactorcraft as eng


@pytest.fixture(scope="module")
def large_panel():
    """Panel with 6 stocks x 100 days for selection tests (needs >5 per cross-section)."""
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


class TestICScreen:
    def test_basic(self, large_panel):
        result = selection.ic_screen(
            large_panel, factor_cols=["mom_5", "mom_10", "value"]
        )
        assert "passed" in result
        assert "failed" in result
        assert "report" in result
        assert len(result["passed"]) + len(result["failed"]) == 3

    def test_strict_threshold(self, large_panel):
        result = selection.ic_screen(
            large_panel, factor_cols=["mom_5", "mom_10", "value"],
            min_abs_ic=0.5, min_ir=2.0
        )
        # With strict thresholds, all should fail
        assert len(result["passed"]) == 0

    def test_lenient_threshold(self, large_panel):
        result = selection.ic_screen(
            large_panel, factor_cols=["mom_5", "mom_10"],
            min_abs_ic=0.0, min_ir=0.0
        )
        assert len(result["passed"]) >= 1

    def test_report_columns(self, large_panel):
        result = selection.ic_screen(large_panel, factor_cols=["mom_5"])
        report = result["report"]
        for col in ["factor", "ic_mean", "ic_std", "ir", "passed"]:
            assert col in report.columns


class TestCorrelationScreen:
    def test_basic(self, large_panel):
        result = selection.correlation_screen(
            large_panel, factor_cols=["mom_5", "mom_10", "value", "size"]
        )
        assert "kept" in result
        assert "removed" in result
        assert "corr_matrix" in result
        assert len(result["kept"]) + len(result["removed"]) == 4

    def test_duplicate_factors(self, large_panel):
        df = large_panel.copy()
        df["dup_mom"] = df["mom_5"]
        result = selection.correlation_screen(
            df, factor_cols=["mom_5", "dup_mom", "value"],
            max_corr=0.99
        )
        assert len(result["removed"]) >= 1

    def test_corr_matrix_shape(self, large_panel):
        cols = ["mom_5", "mom_10", "value"]
        result = selection.correlation_screen(large_panel, factor_cols=cols)
        cm = result["corr_matrix"]
        assert cm.shape == (3, 3)


class TestStabilityScreen:
    def test_basic(self, large_panel):
        result = selection.stability_screen(
            large_panel, factor_cols=["mom_5", "mom_10", "value"]
        )
        assert "factor" in result.columns
        assert "ic_turnover" in result.columns
        assert "stability_rank" in result.columns
        assert len(result) == 3

    def test_rank_order(self, large_panel):
        result = selection.stability_screen(
            large_panel, factor_cols=["mom_5", "mom_10"]
        )
        # At least one factor should have a valid stability rank
        assert result["stability_rank"].notna().any()


class TestSelectTop:
    def test_basic(self, large_panel):
        top = selection.select_top(
            large_panel, factor_cols=["mom_5", "mom_10", "value", "size"],
            top_n=2, criterion="ir"
        )
        assert len(top) == 2
        assert all(isinstance(f, str) for f in top)

    def test_ic_criterion(self, large_panel):
        top = selection.select_top(
            large_panel, factor_cols=["mom_5", "mom_10", "value", "size"],
            top_n=2, criterion="ic"
        )
        assert len(top) == 2

    def test_stability_criterion(self, large_panel):
        top = selection.select_top(
            large_panel, factor_cols=["mom_5", "mom_10", "value", "size"],
            top_n=2, criterion="stability"
        )
        assert len(top) == 2

    def test_top_n_too_large(self, large_panel):
        with pytest.raises(ValueError, match="top_n"):
            selection.select_top(
                large_panel, factor_cols=["mom_5", "mom_10"], top_n=5
            )


class TestFactorReport:
    def test_basic(self, large_panel):
        report = selection.factor_report(
            large_panel, factor_cols=["mom_5", "mom_10", "value"]
        )
        assert len(report) == 3
        assert "factor" in report.columns

    def test_specific_forward_cols(self, large_panel):
        report = selection.factor_report(
            large_panel, factor_cols=["mom_5"],
            forward_cols=["forward_1"]
        )
        assert "forward_1_ic_mean" in report.columns
        assert "forward_1_ir" in report.columns
