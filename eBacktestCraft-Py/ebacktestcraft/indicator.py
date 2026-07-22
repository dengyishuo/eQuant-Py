"""add_indicator — unified cross-package indicator routing.

Single entry point that dispatches indicator computation to the correct
sub-package: eTTR, eClassic, eAlpha101, or eCandleSticks.

Usage::

    from ebacktestcraft import add_indicator

    df = add_indicator(df, "rsi")                # →    ettr.rsi()
    df = add_indicator(df, "sma")                # →    ettr.sma()
    df = add_indicator(df, "momentum")           # → eclassic.momentum()
    df = add_indicator(df, "alpha001")           # → ealpha101.add_alpha001()
    df = add_indicator(df, "doji")               # → ecandlesticks.add_doji()

Name collision disambiguation::

    df = add_indicator(df, "eClassic.sma")       # → eclassic.sma()
    df = add_indicator(df, "eTTR.volatility")    # →    ettr.volatility()

Discover available indicators::

    from ebacktestcraft import list_indicators
    print(list_indicators())
    print(list_indicators("eClassic"))
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Lazy imports — packages are optional
# ---------------------------------------------------------------------------

_ettr = None
_eclassic = None
_ealpha101 = None
_ecandlesticks = None

def _get_ettr():
    global _ettr
    if _ettr is None:
        import ettr as _ettr
    return _ettr

def _get_eclassic():
    global _eclassic
    if _eclassic is None:
        import eclassic as _eclassic
    return _eclassic

def _get_ealpha101():
    global _ealpha101
    if _ealpha101 is None:
        import ealpha101 as _ealpha101
    return _ealpha101

def _get_ecandlesticks():
    global _ecandlesticks
    if _ecandlesticks is None:
        import ecandlesticks as _ecandlesticks
    return _ecandlesticks


# ---------------------------------------------------------------------------
# Routing table: short_name → (pkg, fn, label)
#   pkg  : "ettr" | "eclassic" | "ealpha101" | "ecandlesticks"
#   fn   : actual Python function name (None = same as short_name)
#   label: human-readable label for list_indicators
# ---------------------------------------------------------------------------

_ROUTES: dict[str, tuple[str, Optional[str], str]] = {}

def _r(pkg: str, name: str, fn: Optional[str] = None, label: Optional[str] = None):
    """Register a route."""
    _ROUTES[name] = (pkg, fn, label or name)

_r("eclassic", "benchmark")
_r("eclassic", "beta")
_r("eclassic", "investment")
_r("eclassic", "momentum")
_r("eclassic", "profitability")
_r("eclassic", "ram")
_r("eclassic", "return_",     label="return")
_r("eclassic", "rps")
_r("eclassic", "size")
_r("eclassic", "slope")
_r("eclassic", "value")
_r("eclassic", "benchmark")

# eClassic disambiguation aliases (R-style "eClassic.xxx")
_r("eclassic", "eClassic.sma",        fn="sma",        label="sma (eClassic)")
_r("eclassic", "eClassic.volatility", fn="volatility", label="volatility (eClassic)")
_r("eclassic", "eClassic.momentum",   fn="momentum",   label="momentum (eClassic)")

# ── eTTR ──
for _name in [
    # Trend
    "sma", "ema", "dema", "wma", "hma", "zlema", "alma",
    "evwma", "vwma", "macd", "adx", "gmma", "tdi", "trix",
    "dpo", "vhf", "kst", "po_",
    # Momentum
    "rsi", "cci", "cmo", "tsi", "smi", "wpr",
    "roc", "momentum", "cti", "rvi", "dvi", "stoch", "kdj",
    # Volatility
    "atr", "tr", "bollinger", "keltner", "donchian", "pbands",
    "volatility",
    # Volume
    "obv", "cmf", "vwap", "mfi", "emv", "clv",
    "chaikin_ad", "chaikin_volatility", "williams_ad",
    # Patterns
    "zigzag", "pivots", "sar", "snr",
    # Misc
    "growth", "adj_ratios", "roll_sfm", "aroon",
    "td_setup", "td_countdown", "na_check", "lags",
    "align_with_index", "calculate_performance",
]:
    _r("ettr", _name)

_r("ettr", "ultimate_oscillator")

# eTTR disambiguation aliases
_r("ettr", "eTTR.sma",        fn="sma",        label="sma (eTTR)")
_r("ettr", "eTTR.volatility", fn="volatility", label="volatility (eTTR)")
_r("ettr", "eTTR.momentum",   fn="momentum",   label="momentum (eTTR)")

# ── eAlpha101 ──
for _i in range(1, 102):
    _name = f"alpha{_i:03d}"
    _r("ealpha101", _name, fn=f"add_{_name}")
for _i in range(1, 102):
    _name = f"{_i:03d}"
    _r("ealpha101", _name, fn=f"add_alpha{_name}", label=f"alpha{_name}")

# ── eCandleSticks ──
_csp_names = [
    # TA-Lib 1-bar
    "doji", "hammer", "hanging_man", "inverted_hammer", "shooting_star",
    "marubozu", "closing_marubozu", "belt_hold", "spinning_top",
    "high_wave", "long_legged_doji", "rickshaw_man", "takuri",
    "long_line", "short_line",
    # TA-Lib 2-bar
    "engulfing", "harami", "dark_cloud_cover", "piercing_pattern",
    "counter_attack", "doji_star", "homing_pigeon", "in_neck",
    "on_neck", "matching_low", "separating_lines", "thrusting", "kicking",
    # TA-Lib 3-bar
    "star", "three_white_soldiers", "three_black_crows",
    "three_inside", "three_outside", "three_line_strike",
    "three_methods", "mat_hold", "abandoned_baby",
    "advance_block", "identical_three_crows", "stalled_pattern",
    "stick_sandwich", "three_stars_in_south", "tristar",
    "two_crows", "unique_three_river", "upside_gap_two_crows",
    "gap_side_side_white", "tasuki_gap", "hikkake",
    "xside_gap_three_methods",
    # TA-Lib 4/5-bar
    "conceal_baby_swallow", "breakaway", "ladder_bottom",
    # Custom
    "long_candle", "long_candle_body", "short_candle", "short_candle_body",
    "gap", "inside_day", "outside_day", "stomach",
    "n_higher_close", "n_lower_close",
    "n_long_white_candles", "n_long_black_candles",
    "n_long_white_candle_bodies", "n_long_black_candle_bodies",
    "n_blended",
]
for _name in _csp_names:
    _r("ecandlesticks", _name, fn=f"add_{_name}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def add_indicator(
    df: pd.DataFrame,
    indicator: str,
    **kwargs,
) -> pd.DataFrame:
    """Compute a financial indicator and append it to the DataFrame.

    Routes to the correct package (eTTR / eClassic / eAlpha101 / eCandleSticks)
    based on the short name.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel with ``date`` and ``code`` columns.
    indicator : str
        Short name of the indicator, e.g. ``"rsi"``, ``"momentum"``,
        ``"alpha001"``, ``"doji"``.
        Use ``"eClassic.sma"`` / ``"eTTR.sma"`` to disambiguate name
        collisions.
    **kwargs
        Forwarded to the underlying function (e.g. ``n=14``,
        ``close_col="close"``).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new indicator column(s) appended.

    Raises
    ------
    ValueError
        If *indicator* is not recognised or the required package is not
        installed.

    Examples
    --------
    >>> add_indicator(df, "rsi", close_col="close", n=14)
    >>> add_indicator(df, "momentum", n=20)
    >>> add_indicator(df, "alpha001", close_col="close")
    >>> add_indicator(df, "doji")
    >>> add_indicator(df, "eClassic.volatility", close_col="adjusted", n=20)
    """
    if indicator not in _ROUTES:
        _suggest(indicator)
        raise ValueError(
            f"Unknown indicator: {indicator!r}. "
            f"Use list_indicators() to see available options."
        )

    pkg, fn, _label = _ROUTES[indicator]

    if pkg == "ettr":
        mod = _get_ettr()
    elif pkg == "eclassic":
        mod = _get_eclassic()
    elif pkg == "ealpha101":
        mod = _get_ealpha101()
    elif pkg == "ecandlesticks":
        mod = _get_ecandlesticks()
    else:
        raise ValueError(f"Internal error: unknown package {pkg!r}")

    func = getattr(mod, fn or indicator)
    return func(df, **kwargs)


# ---------------------------------------------------------------------------
# Discovery helper
# ---------------------------------------------------------------------------

def list_indicators(
    package: Optional[str] = None,
) -> Union[pd.DataFrame, list[str]]:
    """List available indicators that can be used with ``add_indicator``.

    Parameters
    ----------
    package : str, optional
        Filter to a specific package:
        ``"ettr"`` | ``"eclassic"`` | ``"ealpha101"`` | ``"ecandlesticks"``.

    Returns
    -------
    pd.DataFrame or list of str
        If *package* is None, returns a DataFrame with columns
        ``indicator``, ``package``, ``label``, ``function``.
        Otherwise returns a simple list of indicator names.
    """
    if package is not None:
        return sorted(k for k, v in _ROUTES.items() if v[0] == package)

    rows = []
    for ind, (pkg, fn, label) in _ROUTES.items():
        rows.append((ind, pkg, label, fn or ind))
    return pd.DataFrame(rows, columns=["indicator", "package", "label", "function"])


# ---------------------------------------------------------------------------
# Fuzzy suggestion
# ---------------------------------------------------------------------------

def _suggest(indicator: str) -> None:
    """Print suggestions for typos (only when running interactively)."""
    import difflib
    candidates = difflib.get_close_matches(indicator, _ROUTES, n=5, cutoff=0.6)
    if candidates:
        print(f"[eBacktestCraft] Did you mean: {', '.join(candidates)}?")
