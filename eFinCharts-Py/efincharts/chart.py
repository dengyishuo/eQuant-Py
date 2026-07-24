"""Core candlestick chart engine built on mplfinance.

Provides the `candlestick()` function — the main entry point for creating
professional financial charts.  Supports:
- Candlestick (OHLC) plots
- Volume sub-chart
- Moving average overlays (SMA, EMA, WMA, etc.)
- Bollinger Bands, SAR, Donchian, Keltner channels
- MACD, RSI panel indicators
- CSP pattern markers
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .theme import (
    COLOR_BBANDS,
    COLOR_MA_FAST,
    COLOR_MA_SLOW,
    COLOR_MACD,
    COLOR_MACD_HIST_DOWN,
    COLOR_MACD_HIST_UP,
    COLOR_MACD_SIGNAL,
    COLOR_RSI_LINE,
    COLOR_RSI_OB,
    COLOR_RSI_OS,
)

class Chart:
    """A financial chart object that wraps mplfinance calls.

    Created by `candlestick()`.  Supports incremental composition via
    `add_*` methods and final rendering via `show()` / `save()`.

    Usage
    -----
    >>> chart = candlestick(df, volume=True, ma=[5, 20])
    >>> chart.add_macd()
    >>> chart.add_rsi()
    >>> chart.show()
    """

    def __init__(self, data: pd.DataFrame, title: str = "", style: str = "efin"):
        self._data = data
        self._title = title
        self._style = style
        self._addplots: list[dict] = []
        self._volume: bool = False
        self._ma: list[int] = []
        self._kwargs: dict[str, Any] = {}
        self._next_panel = 1  # 0 = price, auto-increment for indicators

    # -- public builder methods ------------------------------------------------

    def add_volume(self) -> "Chart":
        """Add volume sub-chart."""
        self._volume = True
        return self

    def add_ma(self, periods: Union[int, list[int]]) -> "Chart":
        """Add moving average line(s)."""
        if isinstance(periods, int):
            periods = [periods]
        self._ma.extend(periods)
        return self

    def add_bbands(
        self,
        period: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
    ) -> "Chart":
        """Add Bollinger Bands overlay."""
        from .overlays import _compute_bbands

        bands = _compute_bbands(self._data, period, nbdevup, nbdevdn)

        self._addplots.extend([
            dict(data=bands["upper"], color=COLOR_BBANDS, width=0.8, alpha=0.6),
            dict(data=bands["middle"], color=COLOR_BBANDS, width=0.8, alpha=0.4, linestyle="--"),
            dict(data=bands["lower"], color=COLOR_BBANDS, width=0.8, alpha=0.6),
        ])
        return self

    def add_sar(self, accel: tuple = (0.02, 0.2)) -> "Chart":
        """Add Parabolic SAR dots overlay."""
        from .overlays import _compute_sar

        sar = _compute_sar(self._data, accel[0], accel[1])

        self._addplots.append(
            dict(data=sar, type="scatter", marker=".", markersize=4, color=COLOR_MA_SLOW)
        )
        return self

    def add_donchian(self, period: int = 20) -> "Chart":
        """Add Donchian channel overlay."""
        from .overlays import _compute_donchian

        ch = _compute_donchian(self._data, period)

        self._addplots.extend([
            dict(data=ch["upper"], color=COLOR_BBANDS, width=0.8, alpha=0.5, linestyle="--"),
            dict(data=ch["mid"], color=COLOR_MA_FAST, width=0.8, alpha=0.4, linestyle=":"),
            dict(data=ch["lower"], color=COLOR_BBANDS, width=0.8, alpha=0.5, linestyle="--"),
        ])
        return self

    def add_keltner(self, period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> "Chart":
        """Add Keltner channel overlay."""
        from .overlays import _compute_keltner

        ch = _compute_keltner(self._data, period, atr_period, multiplier)

        self._addplots.extend([
            dict(data=ch["upper"], color=COLOR_BBANDS, width=0.8, alpha=0.5),
            dict(data=ch["mid"], color=COLOR_MA_FAST, width=0.8, alpha=0.4, linestyle="--"),
            dict(data=ch["lower"], color=COLOR_BBANDS, width=0.8, alpha=0.5),
        ])
        return self

    def add_macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> "Chart":
        """Add MACD indicator panel (returns a separate MACD chart)."""
        from .indicators import _compute_macd

        macd_data = _compute_macd(self._data, fast, slow, signal)
        p = self._next_panel
        self._next_panel += 1

        # Histogram with up/down colors
        hist_data = macd_data["histogram"]
        hist_up = hist_data.where(hist_data >= 0, np.nan)
        hist_down = hist_data.where(hist_data < 0, np.nan)

        self._addplots.extend([
            dict(data=macd_data["macd"], panel=p, color=COLOR_MACD, width=1.2),
            dict(data=macd_data["signal"], panel=p, color=COLOR_MACD_SIGNAL, width=1.0, alpha=0.7),
            dict(data=hist_up, panel=p, type="bar", color=COLOR_MACD_HIST_UP, width=0.7, alpha=0.8),
            dict(data=hist_down, panel=p, type="bar", color=COLOR_MACD_HIST_DOWN, width=0.7, alpha=0.8),
        ])
        return self

    def add_rsi(self, period: int = 14, overbought: float = 70, oversold: float = 30) -> "Chart":
        """Add RSI indicator panel."""
        from .indicators import _compute_rsi

        rsi = _compute_rsi(self._data, period)
        p = self._next_panel
        self._next_panel += 1

        self._addplots.extend([
            dict(data=rsi, panel=p, color=COLOR_RSI_LINE, width=1.2),
            dict(
                data=pd.Series(overbought, index=rsi.index),
                panel=p, color=COLOR_RSI_OB, width=0.6, linestyle="--", alpha=0.5,
            ),
            dict(
                data=pd.Series(oversold, index=rsi.index),
                panel=p, color=COLOR_RSI_OS, width=0.6, linestyle="--", alpha=0.5,
            ),
        ])
        return self

    def add_adx(self, period: int = 14) -> "Chart":
        """Add ADX/DMI indicator panel."""
        from .indicators import _compute_adx

        adx_data = _compute_adx(self._data, period)
        p = self._next_panel
        self._next_panel += 1
        self._addplots.extend([
            dict(data=adx_data["adx"], panel=p, color="#1565c0", width=1.2),
            dict(data=adx_data["+di"], panel=p, color=COLOR_MACD, width=0.8, alpha=0.7),
            dict(data=adx_data["-di"], panel=p, color=COLOR_MACD_SIGNAL, width=0.8, alpha=0.7),
        ])
        return self

    def add_atr(self, period: int = 14) -> "Chart":
        """Add ATR indicator panel."""
        from .indicators import _compute_atr

        atr = _compute_atr(self._data, period)
        p = self._next_panel
        self._next_panel += 1

        self._addplots.append(
            dict(data=atr, panel=p, color=COLOR_RSI_LINE, width=1.2)
        )
        return self

    def add_stoch(self, k_period: int = 14, k_slow: int = 3, d_period: int = 3) -> "Chart":
        """Add Stochastic indicator panel."""
        from .indicators import _compute_stoch

        stoch_data = _compute_stoch(self._data, k_period, k_slow, d_period)
        p = self._next_panel
        self._next_panel += 1
        self._addplots.extend([
            dict(data=stoch_data["%K"], panel=p, color=COLOR_MA_FAST, width=1.0),
            dict(data=stoch_data["%D"], panel=p, color=COLOR_MA_SLOW, width=1.0, alpha=0.7),
        ])
        return self

    def add_cci(self, period: int = 20) -> "Chart":
        """Add CCI indicator panel."""
        from .indicators import _compute_cci

        cci = _compute_cci(self._data, period)
        p = self._next_panel
        self._next_panel += 1

        self._addplots.append(
            dict(data=cci, panel=p, color=COLOR_RSI_LINE, width=1.2)
        )
        return self

    def add_obv(self) -> "Chart":
        """Add OBV indicator panel."""
        from .indicators import _compute_obv

        obv = _compute_obv(self._data)
        p = self._next_panel
        self._next_panel += 1
        self._addplots.append(
            dict(data=obv, panel=p, color=COLOR_MA_FAST, width=1.0)
        )
        return self

    def add_pattern(
        self,
        signals: pd.DataFrame,
        name: str = "pattern",
        bull: bool = True,
    ) -> "Chart":
        """Add CSP (candlestick pattern) markers on the price chart.

        Parameters
        ----------
        signals : pd.DataFrame
            Signals from eCandleSticks-Py detection. Must have a boolean column
            or a 'signal' column, indexed by date.
        name : str
            Display name for the pattern.
        bull : bool
            True for bullish patterns, False for bearish.  Controls marker color.
        """
        from .patterns import _prepare_pattern_markers

        markers = _prepare_pattern_markers(signals, self._data, name, bull)
        if markers is not None:
            self._addplots.extend(markers)
        return self

    # -- rendering -------------------------------------------------------------

    def _build_addplots(self) -> list:
        """Convert internal addplot dicts to mplfinance objects."""
        import mplfinance as mpf

        result = []
        for ap in self._addplots:
            result.append(mpf.make_addplot(**ap))
        return result

    def show(self, show_data: bool = False, **kwargs):
        """Render the chart using matplotlib.

        Parameters
        ----------
        show_data : bool
            If True, print the underlying OHLC data table alongside the chart.
            Default False (chart only).
        **kwargs
            Additional keyword arguments forwarded to ``mpf.plot()``
            (e.g. ``figratio``, ``figscale``, ``savefig``).
        """
        import mplfinance as mpf
        from .theme import get_style

        style = get_style(self._style)

        # Build kwargs for mpf.plot
        plot_kwargs: dict[str, Any] = {
            "type": "candle",
            "style": style,
            "title": self._title,
            "volume": self._volume,
            "addplot": self._build_addplots(),
            "returnfig": "fig" in kwargs or False,
        }

        # MA lines via mplfinance built-in
        if self._ma:
            plot_kwargs["mav"] = tuple(self._ma)

        # Panel config: determine how many panels we need
        max_panel = 0
        for ap in self._addplots:
            p = ap.get("panel", 0) if isinstance(ap, dict) else 0
            if p > max_panel:
                max_panel = p

        if max_panel > 0:
            # panel_ratios: volume (if present) + indicator panels (excludes main price chart)
            panel_ratios = []
            if self._volume:
                panel_ratios.append(1.0)
            for i in range(max_panel):
                panel_ratios.append(1.2)

            plot_kwargs["panel_ratios"] = tuple(panel_ratios)
            plot_kwargs["figratio"] = (12, 4 + len(panel_ratios) * 1.5)

        plot_kwargs.update(kwargs)
        plot_kwargs.update(self._kwargs)

        # Optionally print data table
        if show_data:
            print("--- Chart Data ---")
            print(self._data.head())
            print(f"... {len(self._data)} rows total\n")

        return mpf.plot(self._data, **plot_kwargs)

    def save(self, filename: str, show_data: bool = False, **kwargs):
        """Save chart to file (PNG, PDF, SVG).

        Parameters
        ----------
        filename : str
            Output file path (e.g. "chart.png").
        show_data : bool
            If True, also print the underlying data table. Default False.
        **kwargs
            Additional keyword arguments forwarded to ``mpf.plot()``.
        """
        kwargs["savefig"] = filename
        return self.show(show_data=show_data, **kwargs)

    def get_data(self) -> pd.DataFrame:
        """Return the underlying OHLC data as a DataFrame.

        Returns a copy of the internal data, safe for further manipulation.

        Returns
        -------
        pd.DataFrame
            OHLC(+V) data with DatetimeIndex.

        Examples
        --------
        >>> chart = candlestick(df, volume=True)
        >>> data = chart.get_data()
        >>> print(data.head())
        """
        return self._data.copy()

    def __repr__(self) -> str:
        return (
            f"Chart(title={self._title!r}, "
            f"rows={len(self._data)}, "
            f"addplots={len(self._addplots)}, "
            f"volume={self._volume})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def candlestick(
    data: pd.DataFrame,
    volume: bool = False,
    ma: Optional[Union[int, list[int]]] = None,
    bbands: bool = False,
    sar: bool = False,
    title: str = "",
    style: str = "efin",
    auto_scale: bool = True,
) -> Chart:
    """Create a candlestick chart.

    The single entry-point for all eFinCharts plotting.  Returns a `Chart`
    object that supports chained `.add_*()` calls and `.show()` / `.save()`.

    Parameters
    ----------
    data : pd.DataFrame
        OHLC(+V) data with DatetimeIndex or a date column.  Columns are
        case-insensitive: Open/High/Low/Close(/Volume).
    volume : bool
        Show volume sub-chart.
    ma : int or list of int, optional
        Moving average period(s) to overlay on the price chart.
    bbands : bool
        Add Bollinger Bands (20, 2) overlay.
    sar : bool
        Add Parabolic SAR dots overlay.
    title : str
        Chart title.
    style : str
        Style preset: "efin" (default), "charles", "classic", "yahoo", "nightclouds".
    auto_scale : bool
        Adjust figure dimensions automatically based on content.

    Returns
    -------
    Chart
        A chart object.  Call `.show()` to display, `.save("path.png")` to export.

    Examples
    --------
    >>> from efincharts import candlestick
    >>> import yfinance as yf
    >>> df = yf.download("AAPL", start="2024-01-01")
    >>>
    >>> # Quick chart with volume and MA
    >>> candlestick(df, volume=True, ma=[5, 20, 60]).show()
    >>>
    >>> # Builder pattern with indicators
    >>> (
    ...     candlestick(df, volume=True, title="AAPL")
    ...     .add_ma([20, 60])
    ...     .add_bbands()
    ...     .add_macd()
    ...     .add_rsi()
    ...     .show()
    ... )
    """
    from .utils import prepare_ohlc

    df = prepare_ohlc(data)
    chart = Chart(df, title=title, style=style)

    if volume:
        chart.add_volume()
    if ma is not None:
        chart.add_ma(ma)
    if bbands:
        chart.add_bbands()
    if sar:
        chart.add_sar()

    # Pre-configure reasonable figure size
    if auto_scale:
        nrows = len(df)
        npanels = 1 + int(volume) + sum(1 for ap in chart._addplots if ap.get("panel", 0) > 0)
        chart._kwargs.setdefault("figratio", (12, 4 + npanels * 1.5))
        chart._kwargs.setdefault("figscale", min(2.0, max(0.8, nrows / 120)))

    return chart
