"""A-share code format conversion shared by tushare / akshare / baostock providers.

Users supply bare 6-digit codes (e.g. ``"600000"``); each data source expects
its own suffix/prefix convention, derived from the leading digit.
"""

from __future__ import annotations

_SH_PREFIXES = ("6", "9")
_SZ_PREFIXES = ("0", "2", "3")
_BJ_PREFIXES = ("4", "8")


def exchange_of(bare_code: str) -> str:
    """Return ``"SH"``, ``"SZ"``, or ``"BJ"`` for a bare 6-digit A-share code."""
    digit = bare_code[0]
    if digit in _SH_PREFIXES:
        return "SH"
    if digit in _SZ_PREFIXES:
        return "SZ"
    if digit in _BJ_PREFIXES:
        return "BJ"
    raise ValueError(f"Cannot infer exchange for code: {bare_code}")


def to_tushare(bare_code: str) -> str:
    """``"600000"`` -> ``"600000.SH"``."""
    return f"{bare_code}.{exchange_of(bare_code)}"


def to_baostock(bare_code: str) -> str:
    """``"600000"`` -> ``"sh.600000"``."""
    return f"{exchange_of(bare_code).lower()}.{bare_code}"


def bare(code: str) -> str:
    """Strip any known suffix/prefix, returning the plain 6-digit code."""
    if "." in code:
        left, right = code.split(".", 1)
        return right if left.lower() in ("sh", "sz", "bj") else left
    return code
