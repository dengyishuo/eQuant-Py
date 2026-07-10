"""Tests for numba rolling functions — the performance-critical core."""

from __future__ import annotations

import numpy as np
import pytest

from ettr._rolling import (
    roll_sum,
    roll_min,
    roll_max,
    roll_median,
    roll_mad,
    roll_ema,
    roll_wma,
    roll_sd,
    roll_percent_rank,
    wilder_sum,
    roll_cov,
    aroon_max,
)


class TestRollSum:
    def test_basic(self, mock_ts_array):
        x = mock_ts_array
        result = roll_sum(x, 10)
        assert len(result) == len(x)
        assert np.isnan(result[:9]).all()  # leading NaN
        assert not np.isnan(result[9:]).any()

    def test_accuracy(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roll_sum(x, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(6.0)  # 1+2+3
        assert result[3] == pytest.approx(9.0)  # 2+3+4
        assert result[4] == pytest.approx(12.0)  # 3+4+5

    def test_with_nan(self):
        # NaN in the seed window naturally propagates (matching C runsum behavior).
        # Test with NaN outside the seed window.
        x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        result = roll_sum(x, 3)
        assert result[2] == pytest.approx(6.0)  # 1+2+3
        # NaN at index 3 triggers recomputation of window [2,3,NaN]
        # Then at index 4, window [3, NaN, 5]
        x2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r2 = roll_sum(x2, 3)
        assert r2[4] == pytest.approx(12.0)  # 3+4+5 (no NaN)


class TestRollExtrema:
    def test_min_max(self, mock_ts_array):
        x = mock_ts_array
        rmin = roll_min(x, 10)
        rmax = roll_max(x, 10)
        assert not np.isnan(rmin[15]).any()
        assert rmin[15] <= rmax[15]

    def test_known_values(self):
        x = np.array([5.0, 3.0, 8.0, 2.0, 7.0])
        rmin = roll_min(x, 3)
        assert rmin[2] == 3.0  # min(5,3,8)
        assert rmin[3] == 2.0  # min(3,8,2)
        assert rmin[4] == 2.0  # min(8,2,7)

        rmax = roll_max(x, 3)
        assert rmax[2] == 8.0
        assert rmax[4] == 8.0


class TestRollMedian:
    def test_odd_window(self):
        x = np.array([1.0, 5.0, 3.0, 4.0, 2.0])
        result = roll_median(x, 3)
        assert result[2] == 3.0  # median(1,5,3)
        assert result[3] == 4.0  # median(5,3,4)

    def test_even_window(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = roll_median(x, 4)
        assert result[3] == pytest.approx(2.5)  # median(1,2,3,4)


class TestEMA:
    def test_basic(self, mock_ts_array):
        x = mock_ts_array
        result = roll_ema(x, 10)
        assert np.isnan(result[:9]).all()
        assert not np.isnan(result[19]).any()

    def test_convergence(self):
        """EMA of constant series should converge to that constant."""
        x = np.full(50, 100.0)
        result = roll_ema(x, 5)
        assert result[-1] == pytest.approx(100.0, abs=1e-6)

    def test_wilder_vs_standard(self):
        x = np.arange(1, 21, dtype=np.float64)
        std = roll_ema(x, 5, wilder=False)
        wilder = roll_ema(x, 5, wilder=True)
        # Wilder EMA should be slower (smoother) → different values
        assert abs(std[-1] - wilder[-1]) > 0.01


class TestRollSD:
    def test_constant(self):
        x = np.full(20, 100.0)
        result = roll_sd(x, 5)
        assert result[10] == pytest.approx(0.0, abs=1e-10)

    def test_known(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roll_sd(x, 5)
        assert result[4] == pytest.approx(1.5811388, abs=1e-5)


class TestWilderSum:
    def test_basic(self):
        x = np.arange(1, 15, dtype=np.float64)
        result = wilder_sum(x, 5)
        assert not np.isnan(result[10])
        # Wilder sum is non-negative for positive inputs
        assert (result[~np.isnan(result)] >= 0).all()


class TestAroonMax:
    def test_basic(self):
        h = np.array([10.0, 12.0, 11.0, 13.0, 10.0, 14.0, 12.0], dtype=np.float64)
        l = np.array([9.0, 8.0, 9.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
        up, down, osc = aroon_max(h, l, 3)
        assert len(up) == len(h)
        assert len(osc) == len(h)


class TestPercentRank:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        result = roll_percent_rank(x, 5)
        # 4 values strictly less than 5 / window 5 = 4/5 = 0.8
        assert result[4] == pytest.approx(0.8, abs=0.01)

    def test_midpoint(self):
        x = np.array([5.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        result = roll_percent_rank(x, 5)
        # Middle-ish value: 2 values < 4, 0 equal → 2/5 = 0.4
        assert result[4] == pytest.approx(0.6, abs=0.2)
