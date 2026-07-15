"""Tests for factor timing regime detection and adjustment."""

from __future__ import annotations

import numpy as np
import pytest

from efactorcraft import timing
import eclassic as factor


@pytest.fixture(scope="module")
def panel_with_trend(mock_panel):
    """Panel with enough data for regime detection."""
    import pandas as pd
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=120, freq="B")
    codes = ["AAPL", "MSFT", "GOOG"]
    rows = []
    for i, d in enumerate(dates):
        for c in codes:
            bp = 150.0 if c == "AAPL" else (330.0 if c == "MSFT" else 140.0)
            trend_val = 0.1 if c == "AAPL" else (-0.05 if c == "MSFT" else 0.2)
            close = bp + trend_val * i + np.random.normal(0, bp * 0.02)
            rows.append({
                "date": d, "code": c, "name": c,
                "open": close * 0.99, "high": close * 1.02, "low": close * 0.98,
                "close": round(close, 2), "adjusted": round(close, 2),
                "volume": np.random.uniform(1e6, 1e7),
                "cap": np.random.uniform(1e10, 1e12),
                "bv": np.random.uniform(1e9, 1e11),
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "code"]).reset_index(drop=True)


class TestRegimeDetect:
    def test_basic(self, panel_with_trend):
        result = timing.regime_detect(panel_with_trend, ma_period=60)
        assert "regime" in result.columns
        regimes = result["regime"].dropna().unique()
        valid = {"bull", "bear", "sideways", "unknown"}
        assert all(r in valid for r in regimes)

    def test_valid_categories(self, panel_with_trend):
        result = timing.regime_detect(panel_with_trend, ma_period=30)
        vals = result["regime"].value_counts()
        # Should have some bull/bear/sideways classifications
        categorized = vals.get("bull", 0) + vals.get("bear", 0) + vals.get("sideways", 0)
        assert categorized > 0, "Should classify at least some periods"

    def test_append_false(self, panel_with_trend):
        result = timing.regime_detect(
            panel_with_trend, ma_period=60, append=False
        )
        assert set(result.columns) == {"date", "code", "name", "regime"}

    def test_custom_close_col(self, panel_with_trend):
        df = panel_with_trend.rename(columns={"adjusted": "AdjClose"})
        result = timing.regime_detect(df, close_col="AdjClose", ma_period=60)
        assert "regime" in result.columns

    def test_short_ma(self, panel_with_trend):
        """Shorter MA period should result in fewer 'unknown' classifications."""
        result_short = timing.regime_detect(panel_with_trend, ma_period=5)
        result_long = timing.regime_detect(panel_with_trend, ma_period=60)
        short_unknown = (result_short["regime"] == "unknown").sum()
        long_unknown = (result_long["regime"] == "unknown").sum()
        assert short_unknown <= long_unknown


class TestTrendFilter:
    def test_basic(self, panel_with_trend):
        result = timing.trend_filter(panel_with_trend, ma_period=30)
        assert "trend" in result.columns
        vals = result["trend"].dropna()
        assert len(vals) > 0

    def test_range(self, panel_with_trend):
        result = timing.trend_filter(panel_with_trend, ma_period=30)
        vals = result["trend"].dropna()
        # Trend should be a positive ratio around 1.0
        assert vals.min() > 0


class TestVolFilter:
    def test_basic(self, panel_with_trend):
        result = timing.vol_filter(panel_with_trend, vol_period=20, lookback=100)
        assert "vol_regime" in result.columns
        vals = result["vol_regime"].dropna()
        assert len(vals) > 0

    def test_positive_values(self, panel_with_trend):
        result = timing.vol_filter(panel_with_trend)
        vals = result["vol_regime"].dropna()
        assert (vals >= 0).all()


class TestTimingWeight:
    @pytest.fixture(scope="class")
    def panel_with_regime(self, panel_with_trend):
        return timing.regime_detect(panel_with_trend, ma_period=30)

    def test_basic(self, panel_with_regime):
        df = factor.momentum(panel_with_regime, n=5)
        result = timing.timing_weight(df, factor_col="mom_5")
        assert "timed_mom_5" in result.columns

    def test_bull_bear_difference(self, panel_with_regime):
        df = factor.momentum(panel_with_regime, n=5)
        result = timing.timing_weight(df, factor_col="mom_5")
        bull_mask = result["regime"] == "bull"
        bear_mask = result["regime"] == "bear"
        if bull_mask.any() and bear_mask.any():
            bull_val = result.loc[bull_mask, "timed_mom_5"].abs().mean()
            bear_val = result.loc[bear_mask, "timed_mom_5"].abs().mean()
            # Bear regime factor should be zeroed out (weight=0)
            assert bear_val < 0.01 or bull_val >= bear_val

    def test_custom_weights(self, panel_with_regime):
        df = factor.momentum(panel_with_regime, n=5)
        result = timing.timing_weight(
            df, factor_col="mom_5",
            weights={"bull": 0.8, "bear": 0.2, "sideways": 0.5, "unknown": 0.5}
        )
        assert "timed_mom_5" in result.columns


class TestAdaptiveComposite:
    def test_basic(self, panel_with_trend):
        df = timing.regime_detect(panel_with_trend, ma_period=30)
        df = factor.momentum(df, n=5)
        df = factor.value(df, bv_col="bv", cap_col="cap")
        df = factor.size(df, cap_col="cap")
        result = timing.adaptive_composite(
            df, factor_cols=["mom_5", "value", "size"],
            regime_factor_map={
                "bull": ["mom_5"],
                "bear": ["value", "size"],
                "sideways": ["mom_5", "value"],
                "unknown": ["mom_5"],
            }
        )
        assert "composite_adaptive" in result.columns
        assert result["composite_adaptive"].notna().any()

    def test_with_forward_col(self, panel_with_trend):
        import efactorcraft as eng
        df = timing.regime_detect(panel_with_trend, ma_period=30)
        df = factor.momentum(df, n=5)
        df = factor.value(df, bv_col="bv", cap_col="cap")
        df = eng.add_next_return(df, close_col="adjusted", periods=[5])
        result = timing.adaptive_composite(
            df, factor_cols=["mom_5", "value"],
        )
        assert "composite_adaptive" in result.columns
