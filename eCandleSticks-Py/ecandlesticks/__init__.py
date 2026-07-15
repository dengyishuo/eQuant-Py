"""
eCandleSticks: Japanese candlestick pattern detection.

All detection functions accept a pandas DataFrame with OHLC columns
(case-insensitive: open/high/low/close) and return a DataFrame of
boolean signal columns aligned to the same index.

Example
-------
>>> import pandas as pd
>>> from ecandlesticks import add_doji, add_engulfing
>>> df = pd.read_csv("ohlc.csv", parse_dates=["date"], index_col="date")
>>> print(add_doji(df))
>>> print(add_engulfing(df))
"""

from .patterns import (
    scan_all,
    # 1-bar (TA-Lib)
    add_doji,
    add_hammer,
    add_hanging_man,
    add_inverted_hammer,
    add_shooting_star,
    add_marubozu,
    add_closing_marubozu,
    add_belt_hold,
    add_spinning_top,
    add_high_wave,
    add_long_legged_doji,
    add_rickshaw_man,
    add_takuri,
    add_long_line,
    add_short_line,
    # 2-bar (TA-Lib)
    add_engulfing,
    add_harami,
    add_dark_cloud_cover,
    add_piercing_pattern,
    add_counter_attack,
    add_doji_star,
    add_homing_pigeon,
    add_in_neck,
    add_on_neck,
    add_matching_low,
    add_separating_lines,
    add_thrusting,
    add_kicking,
    # 3-bar (TA-Lib)
    add_star,
    add_three_white_soldiers,
    add_three_black_crows,
    add_three_inside,
    add_three_outside,
    add_three_line_strike,
    add_three_methods,
    add_mat_hold,
    add_abandoned_baby,
    add_advance_block,
    add_identical_three_crows,
    add_stalled_pattern,
    add_stick_sandwich,
    add_three_stars_in_south,
    add_tristar,
    add_two_crows,
    add_unique_three_river,
    add_upside_gap_two_crows,
    add_gap_side_side_white,
    add_tasuki_gap,
    add_hikkake,
    add_xside_gap_three_methods,
    # 4-bar (TA-Lib)
    add_conceal_baby_swallow,
    # 5-bar (TA-Lib)
    add_breakaway,
    add_ladder_bottom,
)

from ._custom import (
    # 1-bar (custom)
    add_long_candle,
    add_long_candle_body,
    add_short_candle,
    add_short_candle_body,
    # 2-bar (custom)
    add_gap,
    add_inside_day,
    add_outside_day,
    add_stomach,
    # N-bar parameterised
    add_n_higher_close,
    add_n_lower_close,
    add_n_long_white_candles,
    add_n_long_black_candles,
    add_n_long_white_candle_bodies,
    add_n_long_black_candle_bodies,
    add_n_blended,
)

__version__ = "0.1.0"

__all__ = [
    # TA-Lib wrappers
    "add_doji", "add_hammer", "add_hanging_man", "add_inverted_hammer",
    "add_shooting_star", "add_marubozu", "add_closing_marubozu",
    "add_belt_hold", "add_spinning_top", "add_high_wave",
    "add_long_legged_doji", "add_rickshaw_man", "add_takuri",
    "add_long_line", "add_short_line",
    "add_engulfing", "add_harami", "add_dark_cloud_cover",
    "add_piercing_pattern", "add_counter_attack", "add_doji_star",
    "add_homing_pigeon", "add_in_neck", "add_on_neck",
    "add_matching_low", "add_separating_lines", "add_thrusting",
    "add_kicking",
    "add_star", "add_three_white_soldiers", "add_three_black_crows",
    "add_three_inside", "add_three_outside",
    "add_three_line_strike", "add_three_methods", "add_mat_hold",
    "add_abandoned_baby", "add_advance_block", "add_identical_three_crows",
    "add_stalled_pattern", "add_stick_sandwich", "add_three_stars_in_south",
    "add_tristar", "add_two_crows", "add_unique_three_river",
    "add_upside_gap_two_crows", "add_gap_side_side_white",
    "add_tasuki_gap", "add_hikkake", "add_xside_gap_three_methods",
    "add_conceal_baby_swallow",
    "add_breakaway", "add_ladder_bottom",
    # custom patterns
    "add_long_candle", "add_long_candle_body",
    "add_short_candle", "add_short_candle_body",
    "add_gap", "add_inside_day", "add_outside_day", "add_stomach",
    "add_n_higher_close", "add_n_lower_close",
    "add_n_long_white_candles", "add_n_long_black_candles",
    "add_n_long_white_candle_bodies", "add_n_long_black_candle_bodies",
    "add_n_blended",
    "scan_all",
]
