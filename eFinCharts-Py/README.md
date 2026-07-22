# eFinCharts — 金融图表可视化

基于 mplfinance 的专业金融图表库，提供 K 线图、技术指标面板、通道叠加和 CSP 形态标记功能。采用 Builder 模式实现链式调用，支持"涨绿跌红"配色方案。

## 在 eQuant-Py 中的角色

作为**可视化层**，eFinCharts 将整个工具链的计算结果以专业图表形式呈现：

- 接收 OHLC 数据 → 绘制标准 K 线图（涨绿跌红）
- 接收 [eCandleSticks](../eCandleSticks-Py/) 的形态信号 → 叠加箭头形态标记
- 内置技术指标计算 → 叠加均线/通道/面板（MACD/RSI/ADX/ATR 等）
- 与 [eBacktestCraft](../eBacktestCraft-Py/) 互补：eFinCharts 侧重行情图表可视化，eBacktestCraft 侧重策略绩效图表

## 架构层级

```
eCandleSticks (信号计算)    OHLC 数据
       │                       │
       └───────────┬───────────┘
                   ▼
          eFinCharts (可视化层)
          ├── chart        → K 线图引擎 (Chart 类 + candlestick 入口)
          ├── overlays     → 价格叠加 (MA/BBands/SAR/Donchian/Keltner)
          ├── indicators   → 指标面板 (MACD/RSI/ADX/ATR/Stoch/CCI/OBV)
          ├── patterns     → CSP 形态标记叠加
          ├── theme        → 主题/配色方案
          └── utils        → 数据规范化/工具函数
```

## 依赖关系

- pandas, numpy, matplotlib, mplfinance
- 与 [eCandleSticks](../eCandleSticks-Py/) 可选配合用于 CSP 形态标记
- 无其他 eQuant 子包强制依赖

## 安装

```bash
pip install -e eFinCharts-Py
```

## 快速开始

```python
import yfinance as yf
from efincharts import candlestick

# 获取数据
df = yf.download("AAPL", start="2024-01-01", auto_adjust=True)

# 1. 一键出图 — 蜡烛图 + 成交量 + 均线
candlestick(df, volume=True, ma=[5, 20, 60], title="AAPL").show()

# 2. Builder 模式 — 通道叠加
(
    candlestick(df, volume=True)
    .add_ma([20, 60])
    .add_bbands(20, 2)
    .add_sar()
    .show()
)

# 3. 多面板 — MACD + RSI + ADX + Stoch
(
    candlestick(df.tail(180), volume=True)
    .add_macd()
    .add_rsi()
    .add_adx()
    .add_stoch()
    .show()
)

# 4. CSP 形态标记 (配合 eCandleSticks)
from ecandlesticks import add_hammer
signals = add_hammer(df)
candlestick(df, volume=True).add_pattern(signals, name="Hammer").show()

# 5. 完整案例：多通道 + 多面板 + 形态
(
    candlestick(df.tail(250), volume=True, title="AAPL 综合分析")
    .add_ma([20, 60, 120])
    .add_bbands(20, 2)
    .add_keltner(20, 10, 2.0)
    .add_macd()
    .add_rsi(14)
    .show()
)

# 6. 保存为图片
candlestick(df, volume=True, ma=[20, 60]).save("chart.png", dpi=150)
```

## API 参考

### 核心入口

| 函数 | 说明 |
|------|------|
| `candlestick(data, volume=False, ma=None, bbands=False, sar=False, title="", style="efin", auto_scale=True)` | 创建 Chart 对象。`data` 支持 DatetimeIndex + OHLC 列（大小写不敏感，自动处理 yfinance MultiIndex），返回 `Chart` 对象 |

### Chart 构建器方法 — 价格图叠加 (Panel 0)

| 方法 | 说明 |
|------|------|
| `.add_volume()` | 添加成交量子图 |
| `.add_ma(periods)` | 移动平均线，`int` 或 `list[int]`，默认 SMA 计算 |
| `.add_bbands(period=20, nbdevup=2, nbdevdn=2)` | 布林带 (中轨虚线 + 上下轨实线) |
| `.add_sar(accel=(0.02, 0.2))` | 抛物线 SAR 点阵 |
| `.add_donchian(period=20)` | 唐奇安通道 (最高最高价 / 最低最低价) |
| `.add_keltner(period=20, atr_period=10, multiplier=2.0)` | 肯特纳通道 (基于 ATR) |

### Chart 构建器方法 — 指标面板 (Panel 1, 2, ...)

面板自动递增，无需手动指定编号。

| 方法 | 说明 | 输出内容 |
|------|------|---------|
| `.add_macd(fast=12, slow=26, signal=9)` | MACD 面板 | DIF 线 + DEA 信号线 + 红绿柱状图 |
| `.add_rsi(period=14, overbought=70, oversold=30)` | RSI 面板 | RSI 曲线 + 70/30 参考线 |
| `.add_adx(period=14)` | ADX/DMI 面板 | ADX + +DI + -DI 三条线 |
| `.add_atr(period=14)` | ATR 面板 | ATR 波动率曲线 |
| `.add_stoch(k_period=14, k_slow=3, d_period=3)` | 随机指标面板 | %K 快线 + %D 慢线 |
| `.add_cci(period=20)` | CCI 面板 | CCI 商品通道指数 |
| `.add_obv()` | OBV 面板 | 能量潮累积曲线 |

### Chart 构建器方法 — 形态标记

| 方法 | 说明 |
|------|------|
| `.add_pattern(signals, name="pattern", bull=True)` | 在价格图上标记形态信号点。`signals` 来自 eCandleSticks 的返回 DataFrame，`bull=True` 为看涨标记（蓝色），`bull=False` 为看跌标记（红色），`bull=None` 为自动着色 |

### Chart 渲染方法

| 方法 | 说明 |
|------|------|
| `.show(**kwargs)` | 使用 matplotlib 显示图表 |
| `.save(filename, **kwargs)` | 保存为文件 (PNG/PDF/SVG) |

### 主题与工具

| 函数 / 常量 | 说明 |
|-------------|------|
| `efin_style` | 预设"涨绿跌红"风格的 mplfinance style dict |
| `make_efin_style(facecolor, gridcolor, gridstyle, gridalpha)` | 自定义风格生成器 |
| `get_style(name)` | 获取命名风格：`"efin"`, `"charles"`, `"classic"`, `"yahoo"`, `"nightclouds"` |
| `prepare_ohlc(data, date_col, volume_col)` | 标准化 OHLC DataFrame（处理列名、MultiIndex、时区等） |
| `make_addplot(data, panel=0, color=None, ...)` | 构建 mplfinance addplot 字典 |
| `make_addplots(*addplot_dicts)` | 批量转换 addplot 字典为 mplfinance 对象 |

### 配色方案

| 常量 | 颜色 | 含义 |
|------|------|------|
| `COLOR_UP` | `#26a69a` | 上涨绿 |
| `COLOR_DOWN` | `#ef5350` | 下跌红 |
| `COLOR_BBANDS` | `#ab47bc` | 布林带紫 |
| `COLOR_MA_FAST` | `#42a5f5` | 快均线蓝 |
| `COLOR_MA_SLOW` | `#ff7043` | 慢均线橙 |
| `COLOR_MACD` | `#1e88e5` | MACD 蓝色 |
| `COLOR_MACD_SIGNAL` | `#e53935` | MACD 信号线红色 |
| `COLOR_MACD_HIST_UP` | `#26a69a` | MACD 红柱 (上涨) |
| `COLOR_MACD_HIST_DOWN` | `#ef5350` | MACD 绿柱 (下跌) |
| `COLOR_RSI_LINE` | `#5c6bc0` | RSI 紫色 |
| `COLOR_RSI_OB` | `#ef5350` | RSI 超买线红色 |
| `COLOR_RSI_OS` | `#26a69a` | RSI 超卖线绿色 |
| `COLOR_PATTERN_BULL` | `#1e88e5` | 看涨形态蓝色 |
| `COLOR_PATTERN_BEAR` | `#e53935` | 看跌形态红色 |

## 数据规范

接受标准 OHLC DataFrame，满足以下要求：

- **索引**：`DatetimeIndex` 或可转换为 DatetimeIndex 的列
- **列名**：`Open`/`High`/`Low`/`Close`/`Volume`（大小写不敏感）
- **yfinance 兼容**：自动处理 `auto_adjust=True` 产生的 MultiIndex 列
- 通过 `prepare_ohlc()` 自动规范化

## 设计原则

- **底层借力 mplfinance**：复用蜡烛图/成交量/均线的成熟渲染能力
- **指标自算**：MACD/RSI/ADX 等技术指标全部纯 pandas/numpy 实现，不依赖 TA-Lib
- **Builder 模式**：链式调用 `.add_*().add_*().show()` 保持 API 简洁直观
- **渐进式复杂度**：从一行 `candlestick(df).show()` 到完整多面板，统一调用语法
- **面板自动管理**：`add_*` 方法自动分配面板编号，自动计算 panel_ratios

## 与各子包的关系

- **eCandleSticks**：`add_pattern()` 方法直接消费 eCandleSticks 返回的信号 DataFrame，将布尔信号转为 K 线图上的方向箭头
- **eTTR**：eFinCharts 内置的计算逻辑与 eTTR 等价但独立实现（不强制依赖 eTTR），如需大量计算建议使用 eTTR
- **eBacktestCraft**：职责互补 — eFinCharts 负责行情图表可视化，eBacktestCraft 负责策略绩效图表
- **webapp**：应用层集成，用于回测结果页面的 K 线展示

## 版本

0.1.0 — Python >= 3.9
