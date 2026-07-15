# eCandleSticks-Py

Japanese candlestick pattern detection for OHLC price series.  
Thin pandas wrapper over [TA-Lib](https://ta-lib.org/) covering all 61 CDL pattern functions.

## Installation

```bash
pip install eCandleSticks          # requires TA-Lib C library installed
```

Or as part of the **eQuant** umbrella:

```bash
pip install eQuant
```

## Quick Start

```python
import pandas as pd
from ecandlesticks import add_engulfing, add_star, scan_all

df = pd.read_csv("ohlc.csv", parse_dates=["date"], index_col="date")

# single pattern
eng = add_engulfing(df)
print(eng[eng.any(axis=1)])
# BullEngulfing  BearEngulfing
# 2024-03-15      True          False
# ...

# all 88 signals at once
all_signals = scan_all(df)
print(all_signals.sum().sort_values(ascending=False).head(10))
```

## API

All functions share the same signature:

```python
add_xxx(df: pd.DataFrame, **kwargs) -> pd.DataFrame
```

- `df` must contain columns `open / high / low / close` (case-insensitive).
- Returns a `pd.DataFrame` of **boolean** columns aligned to `df.index`.
- `scan_all(df)` runs every pattern and concatenates all 88 signal columns.

## Pattern Reference

### 1-bar (15 functions → 28 signals)

| Function | Signals |
|---|---|
| `add_doji` | `Doji`, `DragonflyDoji`, `GravestoneDoji` |
| `add_hammer` | `Hammer` |
| `add_hanging_man` | `HangingMan` |
| `add_inverted_hammer` | `InvertedHammer` |
| `add_shooting_star` | `ShootingStar` |
| `add_marubozu` | `BullMarubozu`, `BearMarubozu` |
| `add_closing_marubozu` | `BullClosingMarubozu`, `BearClosingMarubozu` |
| `add_belt_hold` | `BullBeltHold`, `BearBeltHold` |
| `add_spinning_top` | `BullSpinningTop`, `BearSpinningTop` |
| `add_high_wave` | `BullHighWave`, `BearHighWave` |
| `add_long_legged_doji` | `LongLeggedDoji` |
| `add_rickshaw_man` | `RickshawMan` |
| `add_takuri` | `Takuri` |
| `add_long_line` | `BullLongLine`, `BearLongLine` |
| `add_short_line` | `BullShortLine`, `BearShortLine` |

### 2-bar (12 functions → 22 signals)

| Function | Signals |
|---|---|
| `add_engulfing` | `BullEngulfing`, `BearEngulfing` |
| `add_harami` | `BullHarami`, `BearHarami`, `BullHaramiCross`, `BearHaramiCross` |
| `add_dark_cloud_cover` | `DarkCloudCover` |
| `add_piercing_pattern` | `PiercingPattern` |
| `add_counter_attack` | `BullCounterAttack`, `BearCounterAttack` |
| `add_doji_star` | `BullDojiStar`, `BearDojiStar` |
| `add_homing_pigeon` | `HomingPigeon` |
| `add_in_neck` | `InNeck` |
| `add_on_neck` | `OnNeck` |
| `add_matching_low` | `MatchingLow` |
| `add_separating_lines` | `BullSeparatingLines`, `BearSeparatingLines` |
| `add_thrusting` | `Thrusting` |
| `add_kicking` | `BullKicking`, `BearKicking`, `BullKickingByLength`, `BearKickingByLength` |

### 3-bar (18 functions → 34 signals)

| Function | Signals |
|---|---|
| `add_star` | `MorningStar`, `EveningStar`, `MorningDojiStar`, `EveningDojiStar` |
| `add_three_white_soldiers` | `ThreeWhiteSoldiers` |
| `add_three_black_crows` | `ThreeBlackCrows` |
| `add_three_inside` | `ThreeInsideUp`, `ThreeInsideDown` |
| `add_three_outside` | `ThreeOutsideUp`, `ThreeOutsideDown` |
| `add_three_line_strike` | `BullThreeLineStrike`, `BearThreeLineStrike` |
| `add_three_methods` | `RisingThreeMethods`, `FallingThreeMethods` |
| `add_mat_hold` | `MatHold` |
| `add_abandoned_baby` | `BullAbandonedBaby`, `BearAbandonedBaby` |
| `add_advance_block` | `AdvanceBlock` |
| `add_identical_three_crows` | `Identical3Crows` |
| `add_stalled_pattern` | `StalledPattern` |
| `add_stick_sandwich` | `StickSandwich` |
| `add_three_stars_in_south` | `ThreeStarsInSouth` |
| `add_tristar` | `BullTristar`, `BearTristar` |
| `add_two_crows` | `TwoCrows` |
| `add_unique_three_river` | `Unique3River` |
| `add_upside_gap_two_crows` | `UpsideGap2Crows` |
| `add_gap_side_side_white` | `BullGapSideSideWhite`, `BearGapSideSideWhite` |
| `add_tasuki_gap` | `UpsideTasukiGap`, `DownsideTasukiGap` |
| `add_hikkake` | `BullHikkake`, `BearHikkake`, `BullHikkakeMod`, `BearHikkakeMod` |
| `add_xside_gap_three_methods` | `BullXSideGap3Methods`, `BearXSideGap3Methods` |

### 4-bar

| Function | Signals |
|---|---|
| `add_conceal_baby_swallow` | `ConcealBabySwallow` |

### 5-bar

| Function | Signals |
|---|---|
| `add_breakaway` | `BullBreakaway`, `BearBreakaway` |
| `add_ladder_bottom` | `LadderBottom` |

## Dependencies

- `pandas >= 1.5`
- `numpy >= 1.23`
- `TA-Lib` (requires the native C library: [installation guide](https://ta-lib.org/install/))

## License

GPL-3 © Deng Yishuo

---

## 联系我们

| | |
|---|---|
| 🌐 公司官网 | [xquant.shop](https://xquant.shop) |
| 📱 公司公众号 | xquant-shop |
| 📱 个人公众号 | i锐角 |