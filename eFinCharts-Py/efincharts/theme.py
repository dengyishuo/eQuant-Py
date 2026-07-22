"""mplfinance styles and color themes for financial charting.

Provides pre-configured mplfinance styles matching the R eFinCharts aesthetic:
- 涨绿跌红 (green-up / red-down) Chinese market convention
- Clean, minimal gridlines
- Professional light background
"""

import mplfinance as mpf

# ---------------------------------------------------------------------------
# Color constants (matching R eFinCharts theme_fin)
# ---------------------------------------------------------------------------
COLOR_UP = "#26a69a"     # 涨 green
COLOR_DOWN = "#ef5350"   # 跌 red
COLOR_NEUTRAL = "#78909c"
COLOR_BG = "#f8f9fa"
COLOR_GRID = "#e0e0e0"
COLOR_TEXT = "#37474f"

# Volume colors (半透明 version for better visibility)
COLOR_VOL_UP = "#26a69a"
COLOR_VOL_DOWN = "#ef5350"
COLOR_VOL_ALPHA = 0.6

# Indicator colors
COLOR_MA_FAST = "#42a5f5"   # blue
COLOR_MA_SLOW = "#ff7043"   # orange
COLOR_BBANDS = "#ab47bc"    # purple
COLOR_MACD = "#26a69a"
COLOR_MACD_SIGNAL = "#ef5350"
COLOR_MACD_HIST_UP = "#26a69a"
COLOR_MACD_HIST_DOWN = "#ef5350"
COLOR_RSI_LINE = "#5c6bc0"
COLOR_RSI_OB = "#ef5350"
COLOR_RSI_OS = "#26a69a"

# CSP pattern marker colors
COLOR_PATTERN_BULL = "#1e88e5"
COLOR_PATTERN_BEAR = "#e53935"

# ---------------------------------------------------------------------------
# mplfinance style builders
# ---------------------------------------------------------------------------


def make_efin_style(
    facecolor: str = COLOR_BG,
    gridcolor: str = COLOR_GRID,
    gridstyle: str = "-",
    gridalpha: float = 0.5,
) -> mpf.available_styles:
    """Build a custom mplfinance style matching eFinCharts aesthetic.

    Returns a style dict suitable for mpf.plot(style=...).
    """
    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        marketcolors=mpf.make_marketcolors(
            up=COLOR_UP,
            down=COLOR_DOWN,
            edge="inherit",
            wick="inherit",
            volume={
                "up": COLOR_VOL_UP,
                "down": COLOR_VOL_DOWN,
            },
            alpha=COLOR_VOL_ALPHA,
        ),
        facecolor=facecolor,
        gridcolor=gridcolor,
        gridstyle=gridstyle,
        gridaxis="both",
        y_on_right=False,
        rc={
            "axes.edgecolor": COLOR_GRID,
            "axes.labelcolor": COLOR_TEXT,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "figure.facecolor": facecolor,
        },
    )
    return style


# Pre-built named styles
efin_style = make_efin_style()


def get_style(name: str = "efin"):
    """Get a named style."""
    styles = {
        "efin": efin_style,
        "default": "charles",
        "classic": "classic",
        "yahoo": "yahoo",
        "nightclouds": "nightclouds",
    }
    return styles.get(name, efin_style)
