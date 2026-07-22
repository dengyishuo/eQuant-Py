# eCandleSticks — 日本蜡烛形态识别

基于 TA-Lib 的 K 线形态信号检测库，覆盖 50+ 种经典和自定义形态，返回布尔型信号矩阵，可与 eFinCharts 无缝集成实现形态可视化标注。

## 在 eQuant-Py 中的角色

作为**信号层**，eCandleSticks 专注于计算和识别蜡烛形态信号：

- **形态检测**：输入 OHLC 数据，输出布尔型信号 DataFrame
- **信号供给**：为 [eBacktestCraft](../eBacktestCraft-Py/) 提供交易信号来源（如 engulfing 形态作为反转信号）
- **可视化配合**：与 [eFinCharts](../eFinCharts-Py/) 紧密集成，信号直接通过 `add_pattern()` 叠加到 K 线图上

它独立于因子层（eTTR/eClassic/eAlpha101），从价格形态角度提供补充信号。

## 架构层级

```
eCandleSticks (信号层)
  ├── patterns.py    → TA-Lib CDL 包装器 (40+ 种标准形态)
  ├── _custom.py     → 自定义形态 (长/短蜡烛、缺口、内外日、N 日连续等, 15+ 种)
  └── _utils.py      → 内部 OHLC 工具

         │  计算形态信号 (bool)
         ▼
  eFinCharts.add_pattern(signals, name)
  eBacktestCraft.signal(df, signal_type="...")
```

## 依赖关系

- pandas, numpy
- **TA-Lib C 库** — 必需依赖（brew install ta-lib / apt install ta-lib-dev）
- 无其他 eQuant 子包依赖
- 被 eFinCharts 和 eBacktestCraft 消费

## 安装

```bash
# 先安装 TA-Lib C 库
# macOS:  brew install ta-lib
# Ubuntu: sudo apt install ta-lib-dev

# 安装 Python 包
pip install -e eCandleSticks-Py
```

## 快速开始

```python
from ecandlesticks import (
    add_hammer, add_engulfing, add_doji,
    add_star, add_harami, add_three_white_soldiers,
    add_inside_day, add_outside_day, add_gap,
    scan_all,
)

# 1. 单形态检测
df = add_hammer(ohlc_data)          # 返回含 Hammer 布尔列的 DataFrame
df = add_engulfing(ohlc_data)       # 含 Bull.Engulfing + Bear.Engulfing 列
df = add_doji(ohlc_data)            # 含 Doji 列

# 2. 多形态链式检测
df = add_harami(df)
df = add_star(df)
df = add_three_white_soldiers(df)

# 3. 一键扫描全部已知形态
all_signals = scan_all(ohlc_data)

# 4. 形态信号用于回测
from ebacktestcraft import signal
engulf_signals = signal(df, indicator_cols=["Bull.Engulfing"],
                        signal_type="threshold", threshold=0.5)
```

## API 参考

### TA-Lib 标准形态

所有函数统一签名：`add_xxx(df, **kwargs) -> pd.DataFrame`，返回含布尔列的 DataFrame。

#### 1 根 K 线形态 (1-bar) — 15 个

| 函数 | 形态名称 | 含义 |
|------|---------|------|
| `add_doji(df)` | Doji | 十字星 |
| `add_hammer(df)` | Hammer | 锤子线 |
| `add_hanging_man(df)` | Hanging Man | 上吊线 |
| `add_inverted_hammer(df)` | Inverted Hammer | 倒锤子 |
| `add_shooting_star(df)` | Shooting Star | 射击之星 |
| `add_marubozu(df)` | Marubozu | 光头光脚 |
| `add_closing_marubozu(df)` | Closing Marubozu | 收盘光头光脚 |
| `add_belt_hold(df)` | Belt Hold | 腰带线 |
| `add_spinning_top(df)` | Spinning Top | 纺锤线 |
| `add_high_wave(df)` | High Wave | 长脚十字 |
| `add_long_legged_doji(df)` | Long Legged Doji | 长腿十字星 |
| `add_rickshaw_man(df)` | Rickshaw Man | 黄包车夫 |
| `add_takuri(df)` | Takuri | 探底形态 |
| `add_long_line(df)` | Long Line | 长蜡烛 |
| `add_short_line(df)` | Short Line | 短蜡烛 |

#### 2 根 K 线形态 (2-bar) — 14 个

| 函数 | 形态名称 | 含义 |
|------|---------|------|
| `add_engulfing(df)` | Engulfing | 吞没形态 (Bull/Bear) |
| `add_harami(df)` | Harami | 孕线形态 (Bull/Bear) |
| `add_dark_cloud_cover(df)` | Dark Cloud Cover | 乌云盖顶 |
| `add_piercing_pattern(df)` | Piercing Pattern | 刺透形态 |
| `add_counter_attack(df)` | Counter Attack | 反击线 (Bull/Bear) |
| `add_separating_lines(df)` | Separating Lines | 分离线 (Bull/Bear) |
| `add_doji_star(df)` | Doji Star | 十字启明星/黄昏星 |
| `add_kicking(df)` | Kicking | 踢出线 (Bull/Bear) |
| `add_homing_pigeon(df)` | Homing Pigeon | 家鸽形态 |
| `add_matching_low(df)` | Matching Low | 等低点 |
| `add_on_neck(df)` | On Neck | 颈上线 |
| `add_in_neck(df)` | In Neck | 颈内线 |
| `add_thrusting(df)` | Thrusting | 插入线 |

#### 3 根及以上 K 线形态 (3-bar+) — 22 个

| 函数 | 形态名称 | 含义 |
|------|---------|------|
| `add_star(df)` | Morning/Evening Star | 启明星/黄昏星 |
| `add_three_white_soldiers(df)` | Three White Soldiers | 三个白兵 |
| `add_three_black_crows(df)` | Three Black Crows | 三只乌鸦 |
| `add_three_inside(df)` | Three Inside Up/Down | 内含三法 |
| `add_three_outside(df)` | Three Outside Up/Down | 外扩三法 |
| `add_three_methods(df)` | Rising/Falling Three Methods | 上升/下降三法 |
| `add_three_line_strike(df)` | Three-Line Strike | 三线反击 |
| `add_three_stars_in_south(df)` | Three Stars In South | 南方三星 |
| `add_unique_three_river(df)` | Unique Three River | 独有三川 |
| `add_abandoned_baby(df)` | Abandoned Baby | 弃婴形态 |
| `add_advance_block(df)` | Advance Block | 前进受阻 |
| `add_identical_three_crows(df)` | Identical Three Crows | 三胞胎乌鸦 |
| `add_stalled_pattern(df)` | Stalled Pattern | 停顿形态 |
| `add_stick_sandwich(df)` | Stick Sandwich | 棍夹形态 |
| `add_tristar(df)` | Tristar | 三星形态 |
| `add_two_crows(df)` | Two Crows | 两只乌鸦 |
| `add_upside_gap_two_crows(df)` | Upside Gap Two Crows | 向上跳空两只乌鸦 |
| `add_gap_side_side_white(df)` | Up/Down Gap Side-by-Side | 跳空并列白线 |
| `add_tasuki_gap(df)` | Tasuki Gap | 田足井补空 |
| `add_hikkake(df)` | Hikkake | 引挂形态 |
| `add_mat_hold(df)` | Mat Hold | 铺垫形态 |
| `add_xside_gap_three_methods(df)` | X-Side Gap 3 Methods | 跳空三法 |
| `add_conceal_baby_swallow(df)` | Concealing Baby Swallow | 藏婴吞没 |
| `add_breakaway(df)` | Breakaway | 脱离形态 |
| `add_ladder_bottom(df)` | Ladder Bottom | 梯底形态 |

### 自定义形态

| 函数 | 说明 | 参数 |
|------|------|------|
| `add_long_candle(df, n=20)` | 长蜡烛（实体超过 N 日均值） | `n` |
| `add_long_candle_body(df, n=20)` | 长实体（同上，只看实体长度） | `n` |
| `add_short_candle(df, n=20)` | 短蜡烛（实体低于 N 日均值） | `n` |
| `add_short_candle_body(df, n=20)` | 短实体 | `n` |
| `add_gap(df)` | 跳空缺口 (Up/Down) | — |
| `add_inside_day(df)` | 内含日 | — |
| `add_outside_day(df)` | 外扩日 | — |
| `add_stomach(df)` | 肚腹形态 | — |
| `add_n_higher_close(df, n=3)` | N 日连续收高 | `n` |
| `add_n_lower_close(df, n=3)` | N 日连续收低 | `n` |
| `add_n_long_white_candles(df, n=3)` | N 日连续白蜡烛 | `n` |
| `add_n_long_black_candles(df, n=3)` | N 日连续黑蜡烛 | `n` |
| `add_n_long_white_candle_bodies(df, n=3)` | N 日连续白实体 | `n` |
| `add_n_long_black_candle_bodies(df, n=3)` | N 日连续黑实体 | `n` |
| `add_n_blended(df, n=3)` | N 日混合蜡烛 | `n` |

### 工具

| 函数 | 说明 |
|------|------|
| `scan_all(df)` | 一键扫描全部已知标准形态，返回含所有形态列的 DataFrame |

## 与 eFinCharts 配合

```python
from ecandlesticks import add_engulfing, add_hammer
from efincharts import candlestick

# 计算信号
signals = add_engulfing(ohlc_df)

# 绘制 K 线图 + 形态标记
(
    candlestick(ohlc_df, volume=True)
    .add_pattern(signals, name="Engulfing", bull=None)  # bull=None 自动着色
    .show()
)

# 多形态叠加
(
    candlestick(ohlc_df, volume=True, ma=[20])
    .add_pattern(add_hammer(ohlc_df), name="Hammer", bull=True)
    .add_pattern(add_star(ohlc_df), name="Star", bull=None)
    .add_pattern(add_doji(ohlc_df), name="Doji", bull=None)
    .show()
)
```

## 数据规范

接受标准宽表 OHLC DataFrame：
- 索引为 `DatetimeIndex`
- 列名 `Open`/`High`/`Low`/`Close`（大小写不敏感）
- 返回与原 index 对齐的布尔型 DataFrame

## 与各子包的关系

- **eFinCharts**：`add_pattern()` 直接消费 eCandleSticks 返回的信号 → K 线图形态标记
- **eBacktestCraft**：布尔信号列可直接作为 `signal()` 的 `indicator_cols` 输入，生成交易信号
- **eFactorCraft**：可选作为因子输入，将形态信号纳入因子筛选流程
- **webapp**：信号与回测页面中可选择形态作为交易条件

## 版本

0.1.0 — Python >= 3.9，需要 TA-Lib C 库
