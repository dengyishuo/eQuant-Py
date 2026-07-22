# eQuant-Py

> 统一量化研究工具包 — Python 生态下的 A 股量化策略研发全链路

eQuant-Py 是一套面向 A 股量化研究的 Python 工具集，覆盖从数据基础设施、因子计算、技术指标、形态识别、因子工程、回测评估到可视化的完整流水线。所有子包统一采用**长格式面板 DataFrame** 作为数据交换标准。

---

## 子包总览

| 子包 | 包名 | 层级 | 核心能力 |
|------|------|------|----------|
| [equant](equant/) | `equant` | 入口层 | 统一伞包 + 通用工具（面板验证、装饰器、类型） |
| [edatatools](edatatools/) | `edatatools` | 数据层 | 交易日历、CACS 复权收益、股票池构建 |
| [eTTR](eTTR-Py/) | `ettr` | 因子层 | 技术指标（趋势/动量/波动率/成交量，55+ 个） |
| [eClassic](eClassic-Py/) | `eclassic` | 因子层 | 经典因子（Fama-French：动量/价值/规模/波动率等） |
| [eAlpha101](eAlpha101-Py/) | `ealpha101` | 因子层 | WorldQuant 101 Alpha 因子（完整实现 + 原语） |
| [eFactorCraft](eFactorCraft-Py/) | `efactorcraft` | 因子工程层 | 预处理/中性化/IC 分析/合成/筛选/择时 |
| [eCandleSticks](eCandleSticks-Py/) | `ecandlesticks` | 信号层 | K 线形态识别（TA-Lib，50+ 种形态） |
| [eBacktestCraft](eBacktestCraft-Py/) | `ebacktestcraft` | 策略层 | 事件驱动回测引擎 + 绩效分析 + 参数扫描 |
| [eFinCharts](eFinCharts-Py/) | `efincharts` | 可视化层 | K 线图 + 技术指标面板 + CSP 形态标记 |
| [webapp](webapp/) | — | 应用层 | Streamlit 可视化回测工作台 |

---

## 架构关系

```
                        ┌─────────────────────────────────────────┐
                        │             webapp (应用层)               │
                        │        Streamlit 可视化回测工作台          │
                        └──────────────────┬──────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────────┐
                        │           equant (统一入口)               │
                        │         版本管理 + 通用工具                │
                        └──────────────────┬──────────────────────┘
                                           │
        ┌──────────────┬──────────┬────────┴─────────┬──────────────┐
        │              │          │                  │              │
        ▼              ▼          ▼                  ▼              ▼
  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌────────────┐ ┌──────────┐
  │  信号层    │ │  因子层    │ │  因子工程  │ │  策略层     │ │ 可视化   │
  │eCandleStks│ │ eTTR      │ │eFactorCraft│ │eBacktestCraft│ │eFinCharts│
  │           │ │ eClassic  │ │            │ │           │ │          │
  │           │ │ eAlpha101 │ │            │ │           │ │          │
  └─────┬─────┘ └─────┬─────┘ └─────┬──────┘ └─────┬──────┘ └──────────┘
        │             │             │              │
        └─────────────┴──────┬──────┴──────────────┘
                             │
                    ┌────────┴────────┐
                    │   数据基础设施    │
                    │   edatatools     │
                    │ 日历/收益/股票池  │
                    └─────────────────┘
```

**数据流**：数据层 → 因子层 → 因子工程层 → 策略层；可视化层覆盖全链路；入口层作为统一 API。

---

## 子包调用关系

| 子包 | 上游依赖 (eQuant 内部) | 下游被调用 |
|------|----------------------|-----------|
| `edatatools` | 无 | 所有上层模块 |
| `eTTR` | 无 | eFactorCraft, eBacktestCraft, eFinCharts, webapp |
| `eClassic` | 无 | eFactorCraft, eBacktestCraft, webapp |
| `eAlpha101` | 无 | eFactorCraft, eBacktestCraft, webapp |
| `eFactorCraft` | eTTR, eClassic, eAlpha101 | eBacktestCraft, webapp |
| `eCandleSticks` | 无 | eFinCharts, webapp |
| `eBacktestCraft` | eTTR, eClassic, eAlpha101, eFactorCraft | webapp |
| `eFinCharts` | eCandleSticks（可选） | webapp |
| `equant` | 全部子包 | 用户直接使用 |
| `webapp` | 全部子包 | 终端用户 |

---

## 典型工作流

```python
# 1. 数据准备 — edatatools
from edatatools import date_range, build_universe
dates = date_range("2024-01-01", "2024-12-31", region="CN")
universe = build_universe("2024-01-01", "2024-06-30", univ_type="fixed",
                          fixed_list=["000001", "000002"])

# 2. 数据获取 — eFactorCraft
from efactorcraft import get_data
df = get_data(codes=["000001", "000002"], start="20240101", end="20241231")

# 3. 因子计算 — eTTR + eClassic
from ettr import rsi, macd
from eclassic import momentum, volatility
df = rsi(df, period=14)
df = macd(df, fast=12, slow=26, signal=9)
df = momentum(df, n=252)

# 4. 因子工程 — eFactorCraft
from efactorcraft import winsorize, standardize, ic_analysis, icir_weighted_composite
df = winsorize(df, factor_col=["rsi_14", "momentum_252"], probs=(0.01, 0.99))
df = standardize(df, factor_col=["rsi_14", "momentum_252"])
ic = ic_analysis(df, factor_cols=["rsi_14", "momentum_252"], forward_col="return_5d")
df["composite"] = icir_weighted_composite(df, ["rsi_14", "momentum_252"], forward_col="return_5d")

# 5. 形态识别 — eCandleSticks
from ecandlesticks import add_engulfing, add_hammer
ohlc_df = add_engulfing(ohlc_df)
ohlc_df = add_hammer(ohlc_df)

# 6. 回测 — eBacktestCraft
from ebacktestcraft import Config, run, signal, equal_weight, plot_all
config = Config(start_date="2024-01-01", end_date="2024-12-31", init_capital=1_000_000)
signals = signal(df, indicator_cols=["composite"], signal_type="quantile", top_n=10)
weights = equal_weight(signals, signal_col="composite_signal")
result = run(df, config, weight_col=weights.columns[-1])
plot_all(result, save_dir="charts/")

# 7. 可视化 — eFinCharts
from efincharts import candlestick
candlestick(ohlc_df, volume=True, ma=[5, 20, 60]).show()
```

---

## 安装

### 安装全部（推荐）

```bash
cd eQuant-Py
pip install -e .
```

### 按需安装单个子包

```bash
# 只需要因子库
pip install -e eTTR-Py -e eClassic-Py -e eAlpha101-Py

# 只需要回测
pip install -e eBacktestCraft-Py

# 只需要图表
pip install -e eFinCharts-Py

# 只需要日历/数据
pip install -e edatatools
```

---

## 数据规范

所有子包遵循统一的**长格式面板 DataFrame**：

| date | code | open | high | low | close | volume | factor_cols... |
|------|------|------|------|-----|-------|--------|---------------|
| 2024-01-02 | 000001 | 10.5 | 10.8 | 10.2 | 10.6 | 123456 | ... |
| 2024-01-02 | 000002 | 8.2 | 8.5 | 8.1 | 8.3 | 234567 | ... |

- `date`: 日期列 (datetime)
- `code`: 股票代码 (str)
- OHLCV 列名使用 snake_case 小写
- 因子列名自由定义，通常以 `_N` 后缀表示参数

---

## 要求

- Python >= 3.9
- pandas >= 1.5
- numpy >= 1.23
- TA-Lib C 库（eCandleSticks 需要，其余子包不需要）
