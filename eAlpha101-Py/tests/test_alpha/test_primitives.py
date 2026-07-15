"""Tests for alpha factor primitives."""

from __future__ import annotations

import numpy as np
import pytest

import ealpha101.primitives as alpha


class TestCSRank:
    def test_basic(self):
        x = np.array([1.0, 3.0, 5.0, 2.0])
        r = alpha.cs_rank(x)
        # rank(pct=True, method='average'): [1→0.25, 3→0.75, 5→1.0, 2→0.5]
        assert r[0] == pytest.approx(0.25, abs=0.01)  # 1.0: rank=1/4=0.25
        assert r[2] == pytest.approx(1.0, abs=0.01)   # 5.0: rank=4/4=1.0

    def test_with_nan(self):
        x = np.array([1.0, np.nan, 5.0, 2.0])
        r = alpha.cs_rank(x)
        assert np.isnan(r[1])
        assert not np.isnan(r[0])


class TestTSRank:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        r = alpha.ts_rank(x, 3)
        assert np.isnan(r[0])
        assert np.isnan(r[1])
        assert r[4] == pytest.approx(1.0, abs=0.1)  # max → rank 1


class TestTSMaxMin:
    def test_max(self):
        x = np.array([3.0, 1.0, 5.0, 2.0, 4.0], dtype=np.float64)
        r = alpha.ts_max(x, 3)
        assert r[2] == 5.0
        assert r[3] == 5.0

    def test_min(self):
        x = np.array([3.0, 1.0, 5.0, 2.0, 4.0], dtype=np.float64)
        r = alpha.ts_min(x, 3)
        assert r[2] == 1.0
        assert r[3] == 1.0


class TestTSArgmax:
    def test_basic(self):
        x = np.array([1.0, 5.0, 3.0, 2.0, 4.0], dtype=np.float64)
        r = alpha.ts_argmax(x, 5)
        # Max at index 1 (0-based), so days ago = 5-1-1 = 3
        assert r[4] == 3.0


class TestDecayLinear:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        r = alpha.decay_linear(x, 3)
        # weights: newest=3, mid=2, oldest=1 → [1,2,3] * [1,2,3] / 6
        expected = (1.0 * 1 + 2.0 * 2 + 3.0 * 3) / 6.0  # = 14/6 = 2.333
        assert r[2] == pytest.approx(expected, abs=1e-6)


class TestSignedPower:
    def test_square(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        r = alpha.signedpower(x, 2)
        assert r[0] == pytest.approx(-4.0)  # sign(-2) * |-2|^2 = -4
        assert r[1] == pytest.approx(-1.0)
        assert r[3] == pytest.approx(1.0)
        assert r[4] == pytest.approx(4.0)

    def test_fractional(self):
        x = np.array([-8.0, 8.0])
        r = alpha.signedpower(x, 1 / 3)
        assert r[0] == pytest.approx(-2.0)  # -8^(1/3) = -2
        assert r[1] == pytest.approx(2.0)


class TestCorrelation:
    def test_perfect_positive(self):
        x = np.arange(10, dtype=np.float64)
        y = 2 * x + 1
        r = alpha.correlation(x, y, 10)
        assert r[-1] == pytest.approx(1.0, abs=0.02)


class TestDeltaDelay:
    def test_delta(self):
        x = np.array([1.0, 2.0, 4.0, 7.0, 11.0], dtype=np.float64)
        d = alpha.delta(x, 1)
        assert d[1] == pytest.approx(1.0)
        assert d[2] == pytest.approx(2.0)

    def test_delay(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        d = alpha.delay(x, 1)
        assert np.isnan(d[0])
        assert d[1] == 1.0
        assert d[2] == 2.0
