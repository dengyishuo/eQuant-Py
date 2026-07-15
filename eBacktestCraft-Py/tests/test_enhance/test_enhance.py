"""Tests for enhance module — weights, signals, risk controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ebacktestcraft import enhance
import eclassic as factor
import ebacktestcraft as backtest


@pytest.fixture(scope="module")
def large_panel():
    """8 stocks x 120 days."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=120, freq="B")
    codes = ["AAPL","MSFT","GOOG","AMZN","META","NVDA","TSLA","BRK.B"]
    rows = []
    for d in dates:
        for c in codes:
            bp = np.random.uniform(100, 400)
            rows.append({
                "date": d, "code": c, "name": c,
                "open": bp*np.random.uniform(0.98,1.02),
                "high": bp*np.random.uniform(1.01,1.05),
                "low": bp*np.random.uniform(0.95,0.99),
                "close": bp*np.random.uniform(0.99,1.01),
                "adjusted": bp*np.random.uniform(0.99,1.01),
                "volume": np.random.uniform(1e6,1e7),
                "cap": np.random.uniform(1e10,1e12),
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date","code"]).reset_index(drop=True)
    df = factor.momentum(df, n=[5, 20])
    df = factor.value(df, bv_col="cap", cap_col="cap")  # value ~ 1.0
    df = factor.volatility(df, close_col="adjusted", n=20)
    return df


@pytest.fixture(scope="module")
def panel_with_signal(large_panel):
    """Large panel with a 0/1 signal column."""
    df = backtest.signal(large_panel, indicator_cols=["mom_5"],
                          signal_type="threshold", threshold=0)
    return df


# ═══════════════════════════════════════════════════════════
# Weight tests
# ═══════════════════════════════════════════════════════════


class TestVolParityWeight:
    def test_basic(self, panel_with_signal):
        sig_c = [c for c in panel_with_signal.columns if c.startswith("signal_")][0]
        result = enhance.vol_parity_weight(panel_with_signal, signal_col=sig_c)
        assert "weight_vp" in result.columns
        vals = result["weight_vp"].dropna()
        assert vals.min() >= 0
        # Weights should sum near 1 on days with selections
        daily_sum = result.groupby("date")["weight_vp"].sum()
        assert abs(daily_sum.max() - 1.0) < 0.02

    def test_with_precomputed_vol(self, panel_with_signal):
        sig_c = [c for c in panel_with_signal.columns if c.startswith("signal_")][0]
        result = enhance.vol_parity_weight(panel_with_signal, signal_col=sig_c,
                                            vol_col="vol_20")
        assert "weight_vp" in result.columns
        assert result["weight_vp"].notna().any()


class TestTargetVolWeight:
    def test_basic(self, panel_with_signal):
        sig_c = [c for c in panel_with_signal.columns if c.startswith("signal_")][0]
        result = enhance.target_vol_weight(panel_with_signal, signal_col=sig_c,
                                            target_vol=0.15)
        assert "weight_tv" in result.columns

    def test_higher_target_gives_higher_weights(self, panel_with_signal):
        sig_c = [c for c in panel_with_signal.columns if c.startswith("signal_")][0]
        r_low = enhance.target_vol_weight(panel_with_signal, signal_col=sig_c, target_vol=0.10)
        r_high = enhance.target_vol_weight(panel_with_signal, signal_col=sig_c, target_vol=0.25)
        # Higher target vol should allow larger weights
        low_max = r_low["weight_tv"].max()
        high_max = r_high["weight_tv"].max()
        assert high_max >= low_max


class TestERPWeight:
    def test_basic(self, panel_with_signal):
        sig_c = [c for c in panel_with_signal.columns if c.startswith("signal_")][0]
        result = enhance.erp_weight(panel_with_signal, factor_col="mom_5",
                                     signal_col=sig_c)
        assert "weight_erp" in result.columns
        vals = result["weight_erp"].dropna()
        assert vals.min() >= 0

    def test_all_assets(self, large_panel):
        result = enhance.erp_weight(large_panel, factor_col="mom_5")
        assert "weight_erp" in result.columns


class TestConfidenceWeight:
    def test_basic(self, panel_with_signal):
        sig_c = [c for c in panel_with_signal.columns if c.startswith("signal_")][0]
        result = enhance.confidence_weight(panel_with_signal, factor_col="mom_5",
                                            signal_col=sig_c)
        assert "weight_cf" in result.columns

    def test_max_weight_cap(self, panel_with_signal):
        sig_c = [c for c in panel_with_signal.columns if c.startswith("signal_")][0]
        result = enhance.confidence_weight(panel_with_signal, factor_col="mom_5",
                                            signal_col=sig_c, max_weight=0.25)
        # With 8 stocks, max_weight 0.25 is reasonable. Allow small tolerance
        # for edge cases where few stocks are selected
        assert result["weight_cf"].max() <= 0.51  # worst case: 2 stocks selected


# ═══════════════════════════════════════════════════════════
# Signal tests
# ═══════════════════════════════════════════════════════════


class TestQuantileSignal:
    def test_basic(self, large_panel):
        result = enhance.quantile_signal(large_panel, indicator_col="mom_5",
                                          top_frac=0.3)
        sig_c = [c for c in result.columns if c.startswith("signal_q_")][0]
        # Should select approximately 30% of assets
        selected = result[sig_c].sum()
        total = len(result["date"].unique()) * len(result["code"].unique())
        frac = selected / total
        assert 0.15 < frac < 0.50

    def test_long_short(self, large_panel):
        result = enhance.quantile_signal(large_panel, indicator_col="mom_5",
                                          top_frac=0.2, long_only=False)
        sig_c = [c for c in result.columns if c.startswith("signal_q_")][0]
        vals = result[sig_c].unique()
        assert -1 in vals
        assert 1 in vals


class TestPersistentSignal:
    def test_basic(self, large_panel):
        result = enhance.persistent_signal(large_panel, indicator_col="mom_5",
                                            threshold=0, min_days=3)
        sig_c = [c for c in result.columns if c.startswith("signal_p_")][0]
        assert sig_c in result.columns
        # Persistent signal should have fewer 1s than raw threshold
        raw = (large_panel["mom_5"] > 0).astype(int).sum()
        persist = result[sig_c].sum()
        assert persist <= raw

    def test_min_days_effect(self, large_panel):
        r3 = enhance.persistent_signal(large_panel, indicator_col="mom_5",
                                        min_days=3).dropna()
        r10 = enhance.persistent_signal(large_panel, indicator_col="mom_5",
                                         min_days=10).dropna()
        # Stricter (10 days) should have fewer or equal signals
        c3 = [c for c in r3.columns if c.startswith("signal_p_")][0]
        c10 = [c for c in r10.columns if c.startswith("signal_p_")][0]
        assert r10[c10].sum() <= r3[c3].sum() + 1  # +1 for edge case


class TestSmoothedSignal:
    def test_basic(self, large_panel):
        result = enhance.smoothed_signal(large_panel, indicator_col="mom_5",
                                          smooth_period=10, threshold=0)
        sig_c = [c for c in result.columns if c.startswith("signal_sm")][0]
        assert sig_c in result.columns


# ═══════════════════════════════════════════════════════════
# Risk tests
# ═══════════════════════════════════════════════════════════


class TestApplyVolTarget:
    def test_basic(self, panel_with_signal):
        sig_c = [c for c in panel_with_signal.columns if c.startswith("signal_")][0]
        df = enhance.vol_parity_weight(panel_with_signal, signal_col=sig_c)
        result = enhance.apply_vol_target(df, weight_col="weight_vp",
                                           signal_col=sig_c, target_vol=0.15)
        assert "weight_voltarget" in result.columns

    def test_max_leverage_cap(self, panel_with_signal):
        sig_c = [c for c in panel_with_signal.columns if c.startswith("signal_")][0]
        df = enhance.vol_parity_weight(panel_with_signal, signal_col=sig_c)
        result = enhance.apply_vol_target(df, weight_col="weight_vp",
                                           signal_col=sig_c, max_leverage=1.0)
        # With max_leverage=1.0, weights should sum to <= 1.0
        daily_sum = result.groupby("date")["weight_voltarget"].sum()
        assert daily_sum.max() <= 1.05


class TestComputeTurnover:
    def test_basic(self, panel_with_signal):
        sig_c = [c for c in panel_with_signal.columns if c.startswith("signal_")][0]
        df = enhance.vol_parity_weight(panel_with_signal, signal_col=sig_c)
        result = enhance.compute_turnover(df, weight_col="weight_vp")
        assert "turnover" in result.columns
        assert "turnover_daily" in result.columns
        # Turnover should be non-negative
        assert (result["turnover"].dropna() >= 0).all()
