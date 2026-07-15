"""Tests for backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ebacktestcraft as backtest
import eclassic as factor


class TestConfig:
    def test_default(self):
        cfg = backtest.Config()
        assert cfg.init_capital == 100_000
        assert cfg.rebalance_cycle == "quarterly"

    def test_set_chaining(self):
        cfg = backtest.Config().set(lot_size=200, rebalance_cycle="monthly")
        assert cfg.lot_size == 200
        assert cfg.rebalance_cycle == "monthly"

    def test_invalid_key(self):
        with pytest.raises(ValueError, match="Unknown"):
            backtest.Config().set(nonexistent=123)


class TestSignal:
    def test_threshold(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        result = backtest.signal(
            df, indicator_cols=["mom_5"], signal_type="threshold",
            threshold=0, compare_op=">",
        )
        sig_cols = [c for c in result.columns if c.startswith("signal_")]
        assert len(sig_cols) == 1
        vals = result[sig_cols[0]].dropna()
        assert vals.isin([0, 1]).all()

    def test_constant(self, mock_panel):
        result = backtest.signal(mock_panel, signal_type="constant", constant_value=1)
        sig_cols = [c for c in result.columns if c.startswith("signal_")]
        assert (result[sig_cols[0]] == 1).all()


class TestEqualWeight:
    def test_basic(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        df = backtest.signal(df, indicator_cols=["mom_5"], signal_type="threshold", threshold=0)
        sig_c = [c for c in df.columns if c.startswith("signal_")][0]
        df = backtest.equal_weight(df, signal_col=sig_c)
        wt_c = [c for c in df.columns if c.startswith("weight_equal_")][0]
        # Weights sum to ~1 on days with selection
        daily_sum = df.groupby("date")[wt_c].sum()
        assert abs(daily_sum.max() - 1.0) < 0.01


class TestRunBacktest:
    def test_basic(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        df = backtest.signal(df, indicator_cols=["mom_5"], signal_type="threshold", threshold=0)
        sig_c = [c for c in df.columns if c.startswith("signal_")][0]
        df = backtest.equal_weight(df, signal_col=sig_c)
        wt_c = [c for c in df.columns if c.startswith("weight_equal_")][0]

        cfg = backtest.Config(init_capital=100_000, rebalance_cycle="monthly")
        result = backtest.run(df, config=cfg, weight_col=wt_c)

        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
        assert "nav" in result.equity_curve.columns
        assert "return" in result.equity_curve.columns

    def test_result_summary(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        df = backtest.signal(df, indicator_cols=["mom_5"], signal_type="threshold", threshold=0)
        sig_c = [c for c in df.columns if c.startswith("signal_")][0]
        df = backtest.equal_weight(df, signal_col=sig_c)
        wt_c = [c for c in df.columns if c.startswith("weight_equal_")][0]

        result = backtest.run(df, config=backtest.Config(rebalance_cycle="monthly"),
                             weight_col=wt_c)
        assert "init_capital" in result.summary
        assert "total_return_pct" in result.summary


class TestPerformanceAnalysis:
    def test_metrics(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        df = backtest.signal(df, indicator_cols=["mom_5"], signal_type="threshold", threshold=0)
        sig_c = [c for c in df.columns if c.startswith("signal_")][0]
        df = backtest.equal_weight(df, signal_col=sig_c)
        wt_c = [c for c in df.columns if c.startswith("weight_equal_")][0]

        result = backtest.run(df, config=backtest.Config(rebalance_cycle="monthly"),
                             weight_col=wt_c)
        metrics = backtest.performance_analysis(result.equity_curve)

        for key in ["sharpe_ratio", "max_drawdown_pct", "total_return_pct",
                     "win_rate_pct", "n_days"]:
            assert key in metrics, f"Missing: {key}"


class TestEquityCurveValidation:
    def test_no_negative_nav(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        df = backtest.signal(df, indicator_cols=["mom_5"], signal_type="threshold", threshold=0)
        sig_c = [c for c in df.columns if c.startswith("signal_")][0]
        df = backtest.equal_weight(df, signal_col=sig_c)
        wt_c = [c for c in df.columns if c.startswith("weight_equal_")][0]

        result = backtest.run(df, config=backtest.Config(rebalance_cycle="monthly"),
                             weight_col=wt_c)
        # NAV should always be positive
        assert (result.equity_curve["nav"] > 0).all()
