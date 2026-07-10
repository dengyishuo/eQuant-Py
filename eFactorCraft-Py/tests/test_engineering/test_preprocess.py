"""Tests for factor engineering preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import efactorcraft as eng
import eclassic as factor


class TestWinsorize:
    def test_basic(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        df = eng.winsorize(df, factor_col="mom_5")
        assert "win_mom_5" in df.columns
        # Winsorized values should be clipped — check per-date
        orig = df["mom_5"].dropna()
        win = df["win_mom_5"].dropna()
        # Winsorized range should be ≤ original range
        assert win.max() - win.min() <= orig.max() - orig.min() + 0.01


class TestStandardize:
    def test_basic(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        df = eng.standardize(df, factor_col="mom_5")
        assert "std_mom_5" in df.columns
        # Cross-sectional z-score: mean≈0, std≈1 per date
        by_date = df.groupby("date")["std_mom_5"]
        means = by_date.mean().dropna()
        assert abs(means.mean()) < 0.1  # approximate zero


class TestICAnalysis:
    def test_basic(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        df = eng.add_next_return(df, close_col="adjusted", periods=[1, 5])
        ic = eng.ic_analysis(df, factor_cols=["mom_5"])
        assert "mom_5" in ic
        assert len(ic["mom_5"]) > 0

    def test_ir(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        df = eng.add_next_return(df, close_col="adjusted", periods=[1])
        ic = eng.ic_analysis(df, factor_cols=["mom_5"])
        ir = eng.ir_analysis(ic)
        assert "mom_5" in ir.index


class TestQuantileRank:
    def test_cross(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        df = eng.quantile_rank(df, factor_cols=["mom_5"], type="cross")
        assert "qrank_mom_5" in df.columns
        qr = df["qrank_mom_5"].dropna()
        assert qr.min() >= 0
        assert qr.max() <= 1

    def test_time(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        df = eng.quantile_rank(df, factor_cols=["mom_5"], type="time", n=30)
        assert "qrank_mom_5" in df.columns


class TestFactorPreprocess:
    def test_pipeline(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        df = eng.factor_preprocess(df, factor_col="mom_5")
        assert "full_neu_mom_5" in df.columns
