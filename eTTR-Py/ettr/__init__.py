"""eTTR — Technical trading rules and indicators for long-format panel DataFrames.

Functions take a long-format panel DataFrame (``date``, ``code``, OHLCV columns)
and return the DataFrame with new indicator columns appended.

Submodules
----------
_rolling   : Numba-accelerated rolling-window primitives
trend      : Moving averages, MACD, ADX, KST, ...
momentum   : RSI, CCI, CMO, TSI, SMI, TRIX, ...
volatility : ATR, Bollinger Bands, Keltner Channels, vol estimators
volume     : OBV, CMF, VWAP, MFI, ...
patterns   : ZigZag, Pivots, SAR, SNR
misc       : Aroon, TD Sequential, growth, adj_ratios
"""

from ettr import _rolling

# ── Trend ──
from ettr.trend import (
    adx,
    alma,
    dema,
    dpo,
    ema,
    evwma,
    gmma,
    hma,
    kst,
    macd,
    po_,
    sma,
    tdi,
    trix,
    vhf,
    vwma,
    wma,
    zlema,
)

# ── Momentum ──
from ettr.momentum import (
    cci,
    cmo,
    cti,
    dvi,
    kdj,
    momentum,
    roc,
    rsi,
    rvi,
    smi,
    stoch,
    tsi,
    ultimate_oscillator,
    wpr,
)

# ── Volatility ──
from ettr.volatility import (
    atr,
    bollinger,
    donchian,
    keltner,
    pbands,
    tr,
    volatility,
)

# ── Volume ──
from ettr.volume import (
    chaikin_ad,
    chaikin_volatility,
    clv,
    cmf,
    emv,
    mfi,
    obv,
    vwap,
    williams_ad,
)

# ── Patterns ──
from ettr.patterns import (
    pivots,
    sar,
    snr,
    zigzag,
)

# ── Misc ──
from ettr.misc import (
    adj_ratios,
    align_with_index,
    aroon,
    calculate_performance,
    growth,
    lags,
    na_check,
    roll_sfm,
    td_countdown,
    td_setup,
)

__all__ = [
    # Trend
    "sma", "ema", "dema", "wma", "hma", "zlema", "alma",
    "evwma", "vwma", "macd", "adx", "gmma", "tdi", "trix",
    "dpo", "vhf", "kst", "po_",
    # Momentum
    "rsi", "cci", "cmo", "tsi", "smi", "wpr", "ultimate_oscillator",
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
]
__version__ = "0.1.0"
