"""Tests for panel-aware indicator functions (trend + momentum + volatility)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ettr as indicators


class TestTrendIndicators:
    def test_sma(self, mock_panel):
        df = indicators.sma(mock_panel, n=10)
        assert "SMA_10" in df.columns
        assert df["SMA_10"].notna().any()

    def test_sma_multi_period(self, mock_panel):
        df = indicators.sma(mock_panel, n=[5, 10, 20])
        assert "SMA_5" in df.columns
        assert "SMA_10" in df.columns
        assert "SMA_20" in df.columns

    def test_ema(self, mock_panel):
        df = indicators.ema(mock_panel, n=10)
        assert "EMA_10" in df.columns

    def test_macd(self, mock_panel):
        df = indicators.macd(mock_panel)
        assert "MACD" in df.columns
        assert "MACD_signal" in df.columns
        assert "MACD_hist" in df.columns
        # MACD line should exist for most bars
        assert df["MACD"].notna().sum() > 10

    def test_adx(self, mock_panel):
        df = indicators.adx(mock_panel, n=14)
        assert "ADX_14" in df.columns
        assert "DIp_14" in df.columns

    def test_hma(self, mock_panel):
        df = indicators.hma(mock_panel, n=20)
        assert "HMA_20" in df.columns

    def test_kst(self, mock_panel):
        df = indicators.kst(mock_panel)
        assert "KST" in df.columns
        assert "KST_signal" in df.columns

    def test_append_false(self, mock_panel):
        """append=False returns only id + new columns."""
        df = indicators.sma(mock_panel, n=10, append=False)
        assert set(df.columns) == {"date", "code", "name", "SMA_10"}
        assert len(df) == len(mock_panel)


class TestMomentumIndicators:
    def test_rsi(self, mock_panel):
        df = indicators.rsi(mock_panel, n=14)
        assert "RSI_14" in df.columns
        rsi_vals = df["RSI_14"].dropna()
        assert rsi_vals.min() >= 0
        assert rsi_vals.max() <= 100

    def test_rsi_multi(self, mock_panel):
        df = indicators.rsi(mock_panel, n=[7, 14])
        assert "RSI_7" in df.columns
        assert "RSI_14" in df.columns

    def test_stoch(self, mock_panel):
        df = indicators.stoch(mock_panel)
        assert "Stoch_slowD" in df.columns

    def test_kdj(self, mock_panel):
        df = indicators.kdj(mock_panel)
        assert "KDJ_K" in df.columns
        assert "KDJ_D" in df.columns
        assert "KDJ_J" in df.columns

    def test_cci(self, mock_panel):
        df = indicators.cci(mock_panel, n=20)
        assert "CCI_20" in df.columns

    def test_roc(self, mock_panel):
        df = indicators.roc(mock_panel, n=10)
        assert "ROC_10" in df.columns


class TestVolatilityIndicators:
    def test_atr(self, mock_panel):
        df = indicators.atr(mock_panel, n=14)
        assert "ATR_14" in df.columns

    def test_bollinger(self, mock_panel):
        df = indicators.bollinger(mock_panel)
        assert "BB_middle" in df.columns
        assert "BB_upper" in df.columns
        assert "BB_lower" in df.columns

    def test_volatility_estimators(self, mock_panel):
        for calc in ["close", "parkinson", "garman.klass"]:
            df = indicators.volatility(mock_panel, n=10, calc=calc)
            col = f"Vol_{calc}_10"
            assert col in df.columns
            assert df[col].dropna().min() >= 0  # vol is non-negative


class TestVolumeIndicators:
    def test_obv(self, mock_panel):
        df = indicators.obv(mock_panel)
        assert "OBV" in df.columns
        # OBV is cumulative, should have some non-zero values
        assert (df["OBV"].dropna() != 0).any()

    def test_cmf(self, mock_panel):
        df = indicators.cmf(mock_panel, n=20)
        assert "CMF_20" in df.columns

    def test_vwap(self, mock_panel):
        df = indicators.vwap(mock_panel)
        assert "VWAP" in df.columns


class TestMiscIndicators:
    def test_growth(self, mock_panel):
        df = indicators.growth(mock_panel, n=5)
        assert "growth_5" in df.columns

    def test_aroon(self, mock_panel):
        df = indicators.aroon(mock_panel, n=25)
        assert "Aroon_up" in df.columns
        assert "Aroon_down" in df.columns
        assert "Aroon_osc" in df.columns

    def test_td_setup(self, mock_panel):
        df = indicators.td_setup(mock_panel)
        assert "TD_Setup" in df.columns

    def test_lags(self, mock_panel):
        df = indicators.lags(mock_panel, col="close", n=[1, 3])
        assert "lag_1" in df.columns
        assert "lag_3" in df.columns

    def test_calculate_performance(self, mock_panel):
        single = mock_panel[mock_panel["code"] == "AAPL"].sort_values("date")
        metrics = indicators.calculate_performance(single, close_col="close")
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics
