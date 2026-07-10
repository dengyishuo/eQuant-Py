"""Tests for Tushare provider — without API calls (requires token)."""

from __future__ import annotations

import pytest

from efactorcraft.providers.tushare import get_token, set_token
from efactorcraft import providers


class TestTokenManagement:
    def test_set_get(self):
        set_token("test_token_12345")
        assert get_token() == "test_token_12345"

    def test_no_token_raises(self):
        set_token("")
        with pytest.raises(ValueError):
            providers._call_api("test_endpoint", {}, ["field1"])


class TestImports:
    """Ensure all functions are importable."""
    def test_imports(self):
        from efactorcraft.providers.tushare import (
            get_index_daily,
            get_sw_daily,
            get_trading_date,
            get_info,
            get_capital,
            get_financial_indicators,
        )
        assert callable(get_index_daily)
        assert callable(get_info)
        assert callable(get_trading_date)


class TestGetInfoValidation:
    def test_invalid_ts_code(self):
        with pytest.raises(ValueError, match="stock code"):
            providers.get_info(ts_code="INVALID")

    def test_invalid_market(self):
        with pytest.raises(ValueError, match="market must be"):
            providers.get_info(market="INVALID")

    def test_invalid_is_hs(self):
        with pytest.raises(ValueError, match="is_hs must be"):
            providers.get_info(is_hs="X")
