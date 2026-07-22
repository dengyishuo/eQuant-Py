# eTTR — 技术交易规则与指标

Numba 加速的技术指标计算库，覆盖趋势、动量、波动率、成交量、形态和杂项六大类 55+ 个指标，全部支持长格式面板 DataFrame 输入输出。

## 在 eQuant-Py 中的角色

作为**因子层**的核心主力，eTTR 提供丰富的技术指标因子：

- 为 [eFactorCraft](../eFactorCraft-Py/) 提供因子输入（中性化、合成、筛选的原材料）
- 为 [eBacktestCraft](../eBacktestCraft-Py/) 提供信号基础（如 RSI 超买超卖、MACD 金叉死叉）
- 为 [eFinCharts](../eFinCharts-Py/) 提供指标数据（eFinCharts 内置等价计算，但 eTTR 可大批量预计算）
- 与 [eClassic](../eClassic-Py/) 互补：eTTR 是技术分析因子，eClassic 是学术风险因子

## 架构层级

```
eTTR (因子层 — 技术指标)
  ├── 滚动窗口原语 (_rolling.py)    → Numba JIT 加速
  ├── 面板级包装 (_panel.py)        → 统一长格式输入输出
  ├── 趋势类 (trend.py)            → sma, ema, macd, adx, gmma, ...
  ├── 动量类 (momentum.py)         → rsi, cci, stoch, kdj, roc, ...
  ├── 波动率类 (volatility.py)      → atr, tr, bollinger, keltner, donchian, ...
  ├── 成交量类 (volume.py)          → obv, cmf, vwap, mfi, emv, ...
  ├── 形态类 (patterns.py)          → zigzag, pivots, sar, snr
  └── 杂项 (misc.py)               → growth, aroon, td_setup, td_countdown, ...
```

## 依赖关系

- pandas, numpy, **numba** (JIT 加速)
- 滚动窗口原语由 `_rolling.py` 实现，纯 Numba 加速
- 面板级包装由 `_panel.py` 实现，统一长格式输入输出
- 无其他 eQuant 子包依赖
- 被 eFactorCraft, eBacktestCraft, eFinCharts 消费

## 安装

```bash
pip install -e eTTR-Py
```

## 快速开始

```python
from ettr import rsi, macd, atr, bollinger, kdj, obv, adx, stoch

# 长格式面板 DataFrame: 列含 date, code, open, high, low, close, volume
df = rsi(df, period=14)                              # 附加 rsi_14 列
df = macd(df, fast=12, slow=26, signal=9)            # 附加 macd_12_26_9 等列
df = atr(df, period=14)                              # 附加 atr_14 列
df = bollinger(df, period=20, nbdev=2)               # 附加 bollinger_upper/lower/mid
df = kdj(df, n=9, k_smooth=3, d_smooth=3)            # KDJ 指标
df = obv(df)                                         # OBV 能量潮
df = adx(df, period=14)                              # 附加 adx_14, pdi_14, mdi_14

# 链式计算
df = (
    df
    .pipe(rsi, period=14)
    .pipe(macd, fast=12, slow=26, signal=9)
    .pipe(atr, period=14)
    .pipe(stoch, k_period=14, k_slow=3, d_period=3)
)
```

## API 参考

所有函数统一签名：`func(df, **params) -> pd.DataFrame`，输入输出均为长格式面板 DataFrame。

### 趋势类 (Trend) — 18 个

| 函数 | 说明 | 关键参数 |
|------|------|---------|
| `sma(df, period)` | 简单移动平均 | `period` |
| `ema(df, period)` | 指数移动平均 | `period` |
| `dema(df, period)` | 双指数移动平均 | `period` |
| `wma(df, period)` | 加权移动平均 | `period` |
| `hma(df, period)` | Hull 移动平均 | `period` |
| `zlema(df, period)` | 零延迟 EMA | `period` |
| `alma(df, period)` | Arnaud Legoux MA | `period` |
| `evwma(df, period)` | 弹性成交量加权 MA | `period` |
| `vwma(df, period)` | 成交量加权 MA | `period` |
| `macd(df, fast, slow, signal)` | MACD | `fast=12, slow=26, signal=9` |
| `adx(df, period)` | 平均趋向指数 | `period=14` |
| `gmma(df)` | Guppy 多重 MA | — |
| `tdi(df, rsi_period, ...)` | 交易者动态指数 | `rsi_period=13` |
| `trix(df, period)` | 三重指数平滑 | `period=14` |
| `dpo(df, period)` | 去趋势价格振荡器 | `period=20` |
| `vhf(df, period)` | 垂直水平过滤 | `period=28` |
| `kst(df, ...)` | 确然指标 (Know Sure Thing) | — |
| `po_(df, fast, slow)` | 价格振荡器 | `fast=12, slow=26` |

### 动量类 (Momentum) — 14 个

| 函数 | 说明 | 关键参数 |
|------|------|---------|
| `rsi(df, period)` | 相对强弱指数 (Wilder 平滑) | `period=14` |
| `cci(df, period)` | 商品通道指数 | `period=20` |
| `cmo(df, period)` | Chande 动量振荡器 | `period=14` |
| `tsi(df, ...)` | 真实强度指数 | — |
| `smi(df, ...)` | 随机动量指数 | — |
| `wpr(df, period)` | 威廉指标 %R | `period=14` |
| `ultimate_oscillator(df, ...)` | 终极振荡器 | — |
| `roc(df, period)` | 变化率 | `period=10` |
| `momentum(df, period)` | 动量 | `period=10` |
| `cti(df, period)` | 相关性趋势指标 | `period=10` |
| `rvi(df, period)` | 相对波动率指数 | `period=14` |
| `dvi(df, period)` | 动态动量指数 | `period=14` |
| `stoch(df, k_period, k_slow, d_period)` | 随机振荡器 | `k_period=14` |
| `kdj(df, n, k_smooth, d_smooth)` | KDJ 指标 | `n=9, k_smooth=3, d_smooth=3` |

### 波动率类 (Volatility) — 7 个

| 函数 | 说明 | 关键参数 |
|------|------|---------|
| `atr(df, period)` | 平均真实波幅 | `period=14` |
| `tr(df)` | 真实波幅 (单值) | — |
| `bollinger(df, period, nbdev)` | 布林带 (upper/mid/lower) | `period=20, nbdev=2` |
| `keltner(df, period, atr_period, mult)` | 肯特纳通道 | `period=20, atr_period=10, mult=2` |
| `donchian(df, period)` | 唐奇安通道 (high/mid/low) | `period=20` |
| `pbands(df, period, nbdev)` | 价格百分比带 | `period=20, nbdev=2` |
| `volatility(df, period)` | 历史波动率 (年化) | `period=20` |

### 成交量类 (Volume) — 9 个

| 函数 | 说明 | 关键参数 |
|------|------|---------|
| `obv(df)` | 能量潮 | — |
| `cmf(df, period)` | 蔡金资金流 | `period=20` |
| `vwap(df)` | 成交量加权均价 | — |
| `mfi(df, period)` | 资金流量指标 | `period=14` |
| `emv(df, period)` | 简易波动指标 | `period=14` |
| `clv(df)` | 收盘位置值 | — |
| `chaikin_ad(df)` | 蔡金累计分布 | — |
| `chaikin_volatility(df, period)` | 蔡金波动率 | `period=10` |
| `williams_ad(df)` | 威廉姆斯累积分布 | — |

### 形态类 (Patterns) — 4 个

| 函数 | 说明 | 关键参数 |
|------|------|---------|
| `zigzag(df, threshold)` | 之字转折点 | `threshold=0.05` |
| `pivots(df, period)` | 枢轴点 (Pivot Points) | `period=10` |
| `sar(df, accel_start, accel_max)` | 抛物线 SAR | `accel_start=0.02, accel_max=0.2` |
| `snr(df, period)` | 支撑阻力水平 | `period=20` |

### 杂项 (Misc) — 10 个

`growth`, `adj_ratios`, `roll_sfm`, `aroon`, `td_setup`, `td_countdown`, `na_check`, `lags`, `align_with_index`, `calculate_performance`

## 数据规范

所有函数接受**长格式面板 DataFrame**：

| date | code | open | high | low | close | volume | ... |
|------|------|------|------|-----|-------|--------|-----|

返回附加新因子列的同一格式 DataFrame。因子列命名规则：`{name}_{params}`，如 `rsi_14`、`macd_12_26_9`。

## 性能

底层滚动窗口原语通过 **Numba JIT** 编译加速，大规模面板计算性能优于纯 pandas 实现。

## 与各子包的关系

- **eFactorCraft**：主要上游 — 接收 eTTR 因子进行预处理、中性化、合成、筛选
- **eBacktestCraft**：信号源 — 技术指标可直接作为 `signal()` 的输入
- **eFinCharts**：互补 — eFinCharts 内置等价计算用于实时绘图，eTTR 用于大批量因子预计算
- **eClassic**：互补 — eTTR 是技术分析因子，eClassic 是学术风险/风格因子
- **webapp**：因子选择面板中可勾选 eTTR 指标进行批量计算

## 版本

0.1.0 — Python >= 3.9，需要 numba
