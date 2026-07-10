"""Tests for classic factors."""

from __future__ import annotations

import numpy as np
import pytest

import eclassic as factor


class TestMomentum:
    def test_basic(self, mock_panel):
        df = factor.momentum(mock_panel, n=5)
        assert "mom_5" in df.columns
        assert df["mom_5"].notna().any()

    def test_multi_period(self, mock_panel):
        df = factor.momentum(mock_panel, n=[2, 5, 10, 20])
        for p in [2, 5, 10, 20]:
            assert f"mom_{p}" in df.columns

    def test_discrete(self, mock_panel):
        df_c = factor.momentum(mock_panel, n=5, type="continuous")
        df_d = factor.momentum(mock_panel, n=5, type="discrete")
        # Continuous and discrete returns differ slightly
        assert not df_c["mom_5"].dropna().equals(df_d["mom_5"].dropna())


class TestValue:
    def test_basic(self, mock_panel):
        df = factor.value(mock_panel)
        assert "value" in df.columns
        assert df["value"].notna().all()  # All have valid bv/cap


class TestSize:
    def test_basic(self, mock_panel):
        df = factor.size(mock_panel)
        assert "size" in df.columns
        # log(cap) should be positive for cap > 1
        assert (df["size"] > 0).all()


class TestVolatility:
    def test_basic(self, mock_panel):
        df = factor.volatility(mock_panel, n=20)
        assert "vol_20" in df.columns
        assert df["vol_20"].dropna().min() >= 0


class TestRAM:
    def test_basic(self, mock_panel):
        df = factor.ram(mock_panel, n=30)
        assert "ram_30" in df.columns


class TestRPS:
    def test_basic(self, mock_panel):
        df = factor.rps(mock_panel, n=20)
        assert "rps_20" in df.columns
        # RPS is a rank [0, 1]
        rps_vals = df["rps_20"].dropna()
        assert rps_vals.min() >= 0
        assert rps_vals.max() <= 1


class TestReturn:
    def test_basic(self, mock_panel):
        df = factor.return_(mock_panel, n=1)
        assert "ret_1" in df.columns
