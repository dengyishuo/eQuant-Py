"""
Candlestick pattern detection — thin pandas wrapper over TA-Lib CDL functions.

TA-Lib returns int32 arrays with values -100 (bearish), 0 (no pattern), 100 (bullish).
Each function here converts that to a DataFrame of boolean / signed-integer columns
aligned to the original DataFrame index.

All public functions share the same signature::

    add_xxx(df, **kwargs) -> pd.DataFrame

Parameters
----------
df : pd.DataFrame
    Must contain columns open / high / low / close (case-insensitive).
**kwargs
    Pattern-specific TA-Lib parameters (e.g. penetration).

Returns
-------
pd.DataFrame
    One or more boolean/int columns aligned to ``df.index``.
    Bullish signals are +100 and bearish signals are -100 in the raw column;
    split variants (Bull*/Bear*) expose boolean masks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from ._utils import _ohlc


# ─── internal helpers ────────────────────────────────────────────────────────

def _raw(df: pd.DataFrame, fn_name: str, **kwargs) -> np.ndarray:
    """Call talib.<fn_name> and return the raw int32 array."""
    o, h, l, c = _ohlc(df)
    fn = getattr(talib, fn_name)
    return fn(o.to_numpy(), h.to_numpy(), l.to_numpy(), c.to_numpy(), **kwargs)


def _single(df: pd.DataFrame, fn_name: str, col: str, **kwargs) -> pd.DataFrame:
    """Pattern with a single output column (raw ±100 / 0)."""
    arr = _raw(df, fn_name, **kwargs)
    return pd.DataFrame({col: arr}, index=df.index)


def _bull_bear(df: pd.DataFrame, fn_name: str,
               bull_col: str, bear_col: str, **kwargs) -> pd.DataFrame:
    """Pattern whose TA-Lib output encodes direction: +100 = bull, -100 = bear."""
    arr = _raw(df, fn_name, **kwargs)
    return pd.DataFrame(
        {bull_col: arr == 100, bear_col: arr == -100},
        index=df.index,
    )


def _bull_only(df: pd.DataFrame, fn_name: str, col: str, **kwargs) -> pd.DataFrame:
    """Bullish-only pattern (TA-Lib returns 100 when detected, else 0)."""
    arr = _raw(df, fn_name, **kwargs)
    return pd.DataFrame({col: arr == 100}, index=df.index)


def _bear_only(df: pd.DataFrame, fn_name: str, col: str, **kwargs) -> pd.DataFrame:
    """Bearish-only pattern (TA-Lib returns -100 when detected, else 0)."""
    arr = _raw(df, fn_name, **kwargs)
    return pd.DataFrame({col: arr == -100}, index=df.index)


# ─── 1-bar patterns ──────────────────────────────────────────────────────────

def add_doji(df: pd.DataFrame) -> pd.DataFrame:
    """Doji / Dragonfly Doji / Gravestone Doji.

    Returns
    -------
    DataFrame with columns: Doji, DragonflyDoji, GravestoneDoji (bool)
    """
    doji      = _raw(df, "CDLDOJI")
    dragonfly = _raw(df, "CDLDRAGONFLYDOJI")
    gravestone = _raw(df, "CDLGRAVESTONEDOJI")
    return pd.DataFrame(
        {
            "Doji":           doji != 0,
            "DragonflyDoji":  dragonfly != 0,
            "GravestoneDoji": gravestone != 0,
        },
        index=df.index,
    )


def add_hammer(df: pd.DataFrame) -> pd.DataFrame:
    """Hammer (bullish 1-bar reversal).

    Returns
    -------
    DataFrame with column: Hammer (bool)
    """
    return _bull_only(df, "CDLHAMMER", "Hammer")


def add_hanging_man(df: pd.DataFrame) -> pd.DataFrame:
    """Hanging Man (bearish 1-bar reversal).

    Returns
    -------
    DataFrame with column: HangingMan (bool)
    """
    return _bear_only(df, "CDLHANGINGMAN", "HangingMan")


def add_inverted_hammer(df: pd.DataFrame) -> pd.DataFrame:
    """Inverted Hammer (bullish 1-bar reversal).

    Returns
    -------
    DataFrame with column: InvertedHammer (bool)
    """
    return _bull_only(df, "CDLINVERTEDHAMMER", "InvertedHammer")


def add_shooting_star(df: pd.DataFrame) -> pd.DataFrame:
    """Shooting Star (bearish 1-bar reversal).

    Returns
    -------
    DataFrame with column: ShootingStar (bool)
    """
    return _bear_only(df, "CDLSHOOTINGSTAR", "ShootingStar")


def add_marubozu(df: pd.DataFrame) -> pd.DataFrame:
    """Marubozu — long candle with no (or tiny) shadows.

    Returns
    -------
    DataFrame with columns: BullMarubozu, BearMarubozu (bool)
    """
    return _bull_bear(df, "CDLMARUBOZU", "BullMarubozu", "BearMarubozu")


def add_closing_marubozu(df: pd.DataFrame) -> pd.DataFrame:
    """Closing Marubozu — no shadow on the closing side.

    Returns
    -------
    DataFrame with columns: BullClosingMarubozu, BearClosingMarubozu (bool)
    """
    return _bull_bear(df, "CDLCLOSINGMARUBOZU",
                      "BullClosingMarubozu", "BearClosingMarubozu")


def add_belt_hold(df: pd.DataFrame) -> pd.DataFrame:
    """Belt-Hold — candle that opens at an extreme (no shadow on open side).

    Returns
    -------
    DataFrame with columns: BullBeltHold, BearBeltHold (bool)
    """
    return _bull_bear(df, "CDLBELTHOLD", "BullBeltHold", "BearBeltHold")


def add_spinning_top(df: pd.DataFrame) -> pd.DataFrame:
    """Spinning Top — small body with long shadows on both sides.

    Returns
    -------
    DataFrame with columns: BullSpinningTop, BearSpinningTop (bool)
    """
    return _bull_bear(df, "CDLSPINNINGTOP", "BullSpinningTop", "BearSpinningTop")


def add_high_wave(df: pd.DataFrame) -> pd.DataFrame:
    """High-Wave — very small body with very long shadows.

    Returns
    -------
    DataFrame with columns: BullHighWave, BearHighWave (bool)
    """
    return _bull_bear(df, "CDLHIGHWAVE", "BullHighWave", "BearHighWave")


def add_long_legged_doji(df: pd.DataFrame) -> pd.DataFrame:
    """Long-Legged Doji — doji with long upper and lower shadows.

    Returns
    -------
    DataFrame with column: LongLeggedDoji (bool)
    """
    arr = _raw(df, "CDLLONGLEGGEDDOJI")
    return pd.DataFrame({"LongLeggedDoji": arr != 0}, index=df.index)


def add_rickshaw_man(df: pd.DataFrame) -> pd.DataFrame:
    """Rickshaw Man — doji with body near the candle midpoint.

    Returns
    -------
    DataFrame with column: RickshawMan (bool)
    """
    arr = _raw(df, "CDLRICKSHAWMAN")
    return pd.DataFrame({"RickshawMan": arr != 0}, index=df.index)


def add_takuri(df: pd.DataFrame) -> pd.DataFrame:
    """Takuri — dragonfly doji with very long lower shadow.

    Returns
    -------
    DataFrame with column: Takuri (bool)
    """
    return _bull_only(df, "CDLTAKURI", "Takuri")


def add_long_line(df: pd.DataFrame) -> pd.DataFrame:
    """Long Line Candle.

    Returns
    -------
    DataFrame with columns: BullLongLine, BearLongLine (bool)
    """
    return _bull_bear(df, "CDLLONGLINE", "BullLongLine", "BearLongLine")


def add_short_line(df: pd.DataFrame) -> pd.DataFrame:
    """Short Line Candle.

    Returns
    -------
    DataFrame with columns: BullShortLine, BearShortLine (bool)
    """
    return _bull_bear(df, "CDLSHORTLINE", "BullShortLine", "BearShortLine")


# ─── 2-bar patterns ──────────────────────────────────────────────────────────

def add_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """Engulfing pattern.

    Returns
    -------
    DataFrame with columns: BullEngulfing, BearEngulfing (bool)
    """
    return _bull_bear(df, "CDLENGULFING", "BullEngulfing", "BearEngulfing")


def add_harami(df: pd.DataFrame) -> pd.DataFrame:
    """Harami and Harami Cross.

    Returns
    -------
    DataFrame with columns: BullHarami, BearHarami, BullHaramiCross, BearHaramiCross (bool)
    """
    arr  = _raw(df, "CDLHARAMI")
    arrx = _raw(df, "CDLHARAMICROSS")
    return pd.DataFrame(
        {
            "BullHarami":      arr  == 100,
            "BearHarami":      arr  == -100,
            "BullHaramiCross": arrx == 100,
            "BearHaramiCross": arrx == -100,
        },
        index=df.index,
    )


def add_dark_cloud_cover(df: pd.DataFrame, penetration: float = 0.5) -> pd.DataFrame:
    """Dark Cloud Cover (bearish 2-bar reversal).

    Parameters
    ----------
    penetration : float
        How far the second candle must close into the first. Default 0.5.

    Returns
    -------
    DataFrame with column: DarkCloudCover (bool)
    """
    return _bear_only(df, "CDLDARKCLOUDCOVER", "DarkCloudCover",
                      penetration=penetration)


def add_piercing_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Piercing Pattern (bullish 2-bar reversal).

    Returns
    -------
    DataFrame with column: PiercingPattern (bool)
    """
    return _bull_only(df, "CDLPIERCING", "PiercingPattern")


def add_counter_attack(df: pd.DataFrame) -> pd.DataFrame:
    """Counter Attack — two opposite candles closing at the same level.

    Returns
    -------
    DataFrame with columns: BullCounterAttack, BearCounterAttack (bool)
    """
    return _bull_bear(df, "CDLCOUNTERATTACK",
                      "BullCounterAttack", "BearCounterAttack")


def add_doji_star(df: pd.DataFrame) -> pd.DataFrame:
    """Doji Star — long candle followed by a gap doji.

    Returns
    -------
    DataFrame with columns: BullDojiStar, BearDojiStar (bool)
    """
    return _bull_bear(df, "CDLDOJISTAR", "BullDojiStar", "BearDojiStar")


def add_homing_pigeon(df: pd.DataFrame) -> pd.DataFrame:
    """Homing Pigeon — two black candles, second inside first's body.

    Returns
    -------
    DataFrame with column: HomingPigeon (bool)
    """
    return _bull_only(df, "CDLHOMINGPIGEON", "HomingPigeon")


def add_in_neck(df: pd.DataFrame) -> pd.DataFrame:
    """In-Neck Pattern (bearish continuation).

    Returns
    -------
    DataFrame with column: InNeck (bool)
    """
    return _bear_only(df, "CDLINNECK", "InNeck")


def add_on_neck(df: pd.DataFrame) -> pd.DataFrame:
    """On-Neck Pattern (bearish continuation).

    Returns
    -------
    DataFrame with column: OnNeck (bool)
    """
    return _bear_only(df, "CDLONNECK", "OnNeck")


def add_matching_low(df: pd.DataFrame) -> pd.DataFrame:
    """Matching Low — two black candles closing at the same level.

    Returns
    -------
    DataFrame with column: MatchingLow (bool)
    """
    return _bull_only(df, "CDLMATCHINGLOW", "MatchingLow")


def add_separating_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Separating Lines — continuation, two opposite candles with same open.

    Returns
    -------
    DataFrame with columns: BullSeparatingLines, BearSeparatingLines (bool)
    """
    return _bull_bear(df, "CDLSEPARATINGLINES",
                      "BullSeparatingLines", "BearSeparatingLines")


def add_thrusting(df: pd.DataFrame) -> pd.DataFrame:
    """Thrusting Pattern (bearish continuation).

    Returns
    -------
    DataFrame with column: Thrusting (bool)
    """
    return _bear_only(df, "CDLTHRUSTING", "Thrusting")


def add_kicking(df: pd.DataFrame) -> pd.DataFrame:
    """Kicking — two marubozu candles of opposite color with a gap.

    Returns
    -------
    DataFrame with columns: BullKicking, BearKicking (bool)
    """
    arr  = _raw(df, "CDLKICKING")
    arrb = _raw(df, "CDLKICKINGBYLENGTH")
    return pd.DataFrame(
        {
            "BullKicking":         arr  == 100,
            "BearKicking":         arr  == -100,
            "BullKickingByLength": arrb == 100,
            "BearKickingByLength": arrb == -100,
        },
        index=df.index,
    )


# ─── 3-bar patterns ──────────────────────────────────────────────────────────

def add_star(df: pd.DataFrame, penetration: float = 0.3) -> pd.DataFrame:
    """Morning Star / Evening Star / Morning Doji Star / Evening Doji Star.

    Parameters
    ----------
    penetration : float
        Penetration into first candle's body. Default 0.3.

    Returns
    -------
    DataFrame with columns:
        MorningStar, EveningStar, MorningDojiStar, EveningDojiStar (bool)
    """
    ms  = _raw(df, "CDLMORNINGSTAR",    penetration=penetration)
    es  = _raw(df, "CDLEVENINGSTAR",    penetration=penetration)
    mds = _raw(df, "CDLMORNINGDOJISTAR", penetration=penetration)
    eds = _raw(df, "CDLEVENINGDOJISTAR", penetration=penetration)
    return pd.DataFrame(
        {
            "MorningStar":     ms  == 100,
            "EveningStar":     es  == -100,
            "MorningDojiStar": mds == 100,
            "EveningDojiStar": eds == -100,
        },
        index=df.index,
    )


def add_three_white_soldiers(df: pd.DataFrame) -> pd.DataFrame:
    """Three Advancing White Soldiers (bullish).

    Returns
    -------
    DataFrame with column: ThreeWhiteSoldiers (bool)
    """
    return _bull_only(df, "CDL3WHITESOLDIERS", "ThreeWhiteSoldiers")


def add_three_black_crows(df: pd.DataFrame) -> pd.DataFrame:
    """Three Black Crows (bearish).

    Returns
    -------
    DataFrame with column: ThreeBlackCrows (bool)
    """
    return _bear_only(df, "CDL3BLACKCROWS", "ThreeBlackCrows")


def add_three_inside(df: pd.DataFrame) -> pd.DataFrame:
    """Three Inside Up / Down.

    Returns
    -------
    DataFrame with columns: ThreeInsideUp, ThreeInsideDown (bool)
    """
    arr = _raw(df, "CDL3INSIDE")
    return pd.DataFrame(
        {"ThreeInsideUp": arr == 100, "ThreeInsideDown": arr == -100},
        index=df.index,
    )


def add_three_outside(df: pd.DataFrame) -> pd.DataFrame:
    """Three Outside Up / Down.

    Returns
    -------
    DataFrame with columns: ThreeOutsideUp, ThreeOutsideDown (bool)
    """
    arr = _raw(df, "CDL3OUTSIDE")
    return pd.DataFrame(
        {"ThreeOutsideUp": arr == 100, "ThreeOutsideDown": arr == -100},
        index=df.index,
    )


def add_three_line_strike(df: pd.DataFrame) -> pd.DataFrame:
    """Three-Line Strike.

    Returns
    -------
    DataFrame with columns: BullThreeLineStrike, BearThreeLineStrike (bool)
    """
    return _bull_bear(df, "CDL3LINESTRIKE",
                      "BullThreeLineStrike", "BearThreeLineStrike")


def add_three_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Rising / Falling Three Methods.

    Returns
    -------
    DataFrame with columns: RisingThreeMethods, FallingThreeMethods (bool)
    """
    arr = _raw(df, "CDLRISEFALL3METHODS")
    return pd.DataFrame(
        {"RisingThreeMethods": arr == 100, "FallingThreeMethods": arr == -100},
        index=df.index,
    )


def add_mat_hold(df: pd.DataFrame, penetration: float = 0.5) -> pd.DataFrame:
    """Mat Hold (bullish continuation).

    Returns
    -------
    DataFrame with column: MatHold (bool)
    """
    return _bull_only(df, "CDLMATHOLD", "MatHold", penetration=penetration)


def add_abandoned_baby(df: pd.DataFrame, penetration: float = 0.3) -> pd.DataFrame:
    """Abandoned Baby — isolated doji with full gaps on both sides.

    Parameters
    ----------
    penetration : float
        Default 0.3.

    Returns
    -------
    DataFrame with columns: BullAbandonedBaby, BearAbandonedBaby (bool)
    """
    return _bull_bear(df, "CDLABANDONEDBABY",
                      "BullAbandonedBaby", "BearAbandonedBaby",
                      penetration=penetration)


def add_advance_block(df: pd.DataFrame) -> pd.DataFrame:
    """Advance Block — three white soldiers with weakening bodies.

    Returns
    -------
    DataFrame with column: AdvanceBlock (bool)
    """
    return _bear_only(df, "CDLADVANCEBLOCK", "AdvanceBlock")


def add_identical_three_crows(df: pd.DataFrame) -> pd.DataFrame:
    """Identical Three Crows.

    Returns
    -------
    DataFrame with column: Identical3Crows (bool)
    """
    return _bear_only(df, "CDLIDENTICAL3CROWS", "Identical3Crows")


def add_stalled_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Stalled Pattern — three white soldiers, last one small.

    Returns
    -------
    DataFrame with column: StalledPattern (bool)
    """
    return _bear_only(df, "CDLSTALLEDPATTERN", "StalledPattern")


def add_stick_sandwich(df: pd.DataFrame) -> pd.DataFrame:
    """Stick Sandwich — black, white, black with equal outer closes.

    Returns
    -------
    DataFrame with column: StickSandwich (bool)
    """
    return _bull_only(df, "CDLSTICKSANDWICH", "StickSandwich")


def add_three_stars_in_south(df: pd.DataFrame) -> pd.DataFrame:
    """Three Stars in the South (bullish).

    Returns
    -------
    DataFrame with column: ThreeStarsInSouth (bool)
    """
    return _bull_only(df, "CDL3STARSINSOUTH", "ThreeStarsInSouth")


def add_tristar(df: pd.DataFrame) -> pd.DataFrame:
    """Tristar — three consecutive doji candles.

    Returns
    -------
    DataFrame with columns: BullTristar, BearTristar (bool)
    """
    return _bull_bear(df, "CDLTRISTAR", "BullTristar", "BearTristar")


def add_two_crows(df: pd.DataFrame) -> pd.DataFrame:
    """Two Crows (bearish).

    Returns
    -------
    DataFrame with column: TwoCrows (bool)
    """
    return _bear_only(df, "CDL2CROWS", "TwoCrows")


def add_unique_three_river(df: pd.DataFrame) -> pd.DataFrame:
    """Unique Three River (bullish).

    Returns
    -------
    DataFrame with column: Unique3River (bool)
    """
    return _bull_only(df, "CDLUNIQUE3RIVER", "Unique3River")


def add_upside_gap_two_crows(df: pd.DataFrame) -> pd.DataFrame:
    """Upside Gap Two Crows (bearish).

    Returns
    -------
    DataFrame with column: UpsideGap2Crows (bool)
    """
    return _bear_only(df, "CDLUPSIDEGAP2CROWS", "UpsideGap2Crows")


def add_gap_side_side_white(df: pd.DataFrame) -> pd.DataFrame:
    """Up/Down-Gap Side-by-Side White Lines.

    Returns
    -------
    DataFrame with columns: BullGapSideSideWhite, BearGapSideSideWhite (bool)
    """
    return _bull_bear(df, "CDLGAPSIDESIDEWHITE",
                      "BullGapSideSideWhite", "BearGapSideSideWhite")


def add_tasuki_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Tasuki Gap (continuation pattern).

    Returns
    -------
    DataFrame with columns: UpsideTasukiGap, DownsideTasukiGap (bool)
    """
    arr = _raw(df, "CDLTASUKIGAP")
    return pd.DataFrame(
        {"UpsideTasukiGap": arr == 100, "DownsideTasukiGap": arr == -100},
        index=df.index,
    )


def add_hikkake(df: pd.DataFrame) -> pd.DataFrame:
    """Hikkake and Modified Hikkake — inside-bar false-breakout trap.

    Returns
    -------
    DataFrame with columns:
        BullHikkake, BearHikkake, BullHikkakeMod, BearHikkakeMod (bool)
    """
    arr  = _raw(df, "CDLHIKKAKE")
    arrm = _raw(df, "CDLHIKKAKEMOD")
    return pd.DataFrame(
        {
            "BullHikkake":    arr  == 100,
            "BearHikkake":    arr  == -100,
            "BullHikkakeMod": arrm == 100,
            "BearHikkakeMod": arrm == -100,
        },
        index=df.index,
    )


def add_xside_gap_three_methods(df: pd.DataFrame) -> pd.DataFrame:
    """X-Side Gap Three Methods.

    Returns
    -------
    DataFrame with columns: BullXSideGap3Methods, BearXSideGap3Methods (bool)
    """
    return _bull_bear(df, "CDLXSIDEGAP3METHODS",
                      "BullXSideGap3Methods", "BearXSideGap3Methods")


# ─── 4-bar patterns ──────────────────────────────────────────────────────────

def add_conceal_baby_swallow(df: pd.DataFrame) -> pd.DataFrame:
    """Concealing Baby Swallow — four black candles, last engulfs third.

    Returns
    -------
    DataFrame with column: ConcealBabySwallow (bool)
    """
    return _bull_only(df, "CDLCONCEALBABYSWALL", "ConcealBabySwallow")


# ─── 5-bar patterns ──────────────────────────────────────────────────────────

def add_breakaway(df: pd.DataFrame) -> pd.DataFrame:
    """Breakaway — gap continuation then reversal into the gap.

    Returns
    -------
    DataFrame with columns: BullBreakaway, BearBreakaway (bool)
    """
    return _bull_bear(df, "CDLBREAKAWAY", "BullBreakaway", "BearBreakaway")


def add_ladder_bottom(df: pd.DataFrame) -> pd.DataFrame:
    """Ladder Bottom (bullish).

    Returns
    -------
    DataFrame with column: LadderBottom (bool)
    """
    return _bull_only(df, "CDLLADDERBOTTOM", "LadderBottom")


# ─── convenience: scan all patterns at once ──────────────────────────────────

def scan_all(df: pd.DataFrame) -> pd.DataFrame:
    """Run every pattern function and concatenate results.

    Includes all TA-Lib wrappers plus the custom pandas/numpy patterns
    (Gap, InsideDay, OutsideDay, Stomach, Long/ShortCandle, N-bar series).
    N-bar patterns use their default parameter values (n=3, lookback=20).

    Returns
    -------
    DataFrame with one boolean column per signal, aligned to ``df.index``.
    """
    from ._custom import (
        add_long_candle, add_long_candle_body,
        add_short_candle, add_short_candle_body,
        add_gap, add_inside_day, add_outside_day, add_stomach,
        add_n_higher_close, add_n_lower_close,
        add_n_long_white_candles, add_n_long_black_candles,
        add_n_long_white_candle_bodies, add_n_long_black_candle_bodies,
        add_n_blended,
    )

    talib_fns = [
        add_doji, add_hammer, add_hanging_man, add_inverted_hammer,
        add_shooting_star, add_marubozu, add_closing_marubozu,
        add_belt_hold, add_spinning_top, add_high_wave,
        add_long_legged_doji, add_rickshaw_man, add_takuri,
        add_long_line, add_short_line,
        add_engulfing, add_harami, add_dark_cloud_cover,
        add_piercing_pattern, add_counter_attack, add_doji_star,
        add_homing_pigeon, add_in_neck, add_on_neck,
        add_matching_low, add_separating_lines, add_thrusting, add_kicking,
        add_star, add_three_white_soldiers, add_three_black_crows,
        add_three_inside, add_three_outside, add_three_line_strike,
        add_three_methods, add_mat_hold, add_abandoned_baby,
        add_advance_block, add_identical_three_crows, add_stalled_pattern,
        add_stick_sandwich, add_three_stars_in_south, add_tristar,
        add_two_crows, add_unique_three_river, add_upside_gap_two_crows,
        add_gap_side_side_white, add_tasuki_gap, add_hikkake,
        add_xside_gap_three_methods,
        add_conceal_baby_swallow,
        add_breakaway, add_ladder_bottom,
    ]
    custom_fns = [
        add_long_candle, add_long_candle_body,
        add_short_candle, add_short_candle_body,
        add_gap, add_inside_day, add_outside_day, add_stomach,
        add_n_higher_close, add_n_lower_close,
        add_n_long_white_candles, add_n_long_black_candles,
        add_n_long_white_candle_bodies, add_n_long_black_candle_bodies,
        add_n_blended,
    ]
    parts = [fn(df) for fn in talib_fns] + [fn(df) for fn in custom_fns]
    return pd.concat(parts, axis=1)
