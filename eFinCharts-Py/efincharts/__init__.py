"""eFinCharts — Professional financial charting for quantitative research.

Built on mplfinance and matplotlib.  Provides:
- Candlestick (K-line) charts with volume and moving average overlays
- Technical indicator panels (MACD, RSI, ADX, ATR, Stochastic, CCI, OBV, etc.)
- CSP (Candlestick Pattern) marker overlays via eCandleSticks-Py integration
- Pre-configured "涨绿跌红" theme for Chinese A-share markets

Quickstart
----------
>>> from efincharts import candlestick
>>> import yfinance as yf
>>> df = yf.download("AAPL", start="2024-01-01")
>>> candlestick(df, volume=True, ma=[5, 20, 60]).show()

>>> # Builder pattern
>>> (
...     candlestick(df, volume=True, title="AAPL")
...     .add_bbands()
...     .add_macd()
...     .add_rsi()
...     .show()
... )
"""

from .chart import Chart, candlestick
from .theme import (
    COLOR_DOWN,
    COLOR_UP,
    efin_style,
    get_style,
    make_efin_style,
)
from .utils import make_addplot, make_addplots, prepare_ohlc

__all__ = [
    # Core
    "candlestick",
    "Chart",
    # Theme
    "efin_style",
    "make_efin_style",
    "get_style",
    "COLOR_UP",
    "COLOR_DOWN",
    # Utilities
    "prepare_ohlc",
    "make_addplot",
    "make_addplots",
]
