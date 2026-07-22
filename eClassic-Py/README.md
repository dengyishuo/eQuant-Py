# eClassic — 经典量化因子

Fama-French / Barra 风格的经典因子库，覆盖动量、价值、规模、波动率、盈利能力、投资等 13 个因子，所有函数支持长格式面板 DataFrame。

## 在 eQuant-Py 中的角色

作为**因子层**的成员，eClassic 提供学术界广泛使用的经典风格因子：

- 与 [eTTR](../eTTR-Py/) 互补：eTTR 是技术分析因子（价格形态与趋势），eClassic 是学术风险因子（风险定价与风格溢价）
- 与 [eAlpha101](../eAlpha101-Py/) 互补：eAlpha101 是 WorldQuant 的公式化 Alpha，eClassic 是 Fama-French / Barra 体系中定义的标准化因子
- 为 [eFactorCraft](../eFactorCraft-Py/) 提供因子输入
- 为 [eBacktestCraft](../eBacktestCraft-Py/) 提供风险归因和风格分析因子

## 架构层级

```
eClassic (因子层 — 经典因子)
  ├── momentum       → 动量因子 (Jegadeesh-Titman)
  ├── value          → 价值因子 (B/M, E/P)
  ├── size           → 规模因子 (log 市值)
  ├── beta           → 贝塔因子 (滚动市场 Beta)
  ├── volatility     → 波动率因子 (标准差/波动率)
  ├── profitability  → 盈利能力因子 (ROE)
  ├── investment     → 投资因子 (资产增长率)
  ├── slope          → 斜率因子 (趋势强度)
  ├── sma            → 均线偏离因子
  ├── rps            → 相对强度 RPS
  ├── ram            → 风险调整动量 (RAM)
  ├── return_        → 收益因子 (合成/真实)
  └── benchmark      → 基准相对收益
```

## 依赖关系

- 仅依赖 pandas 和 numpy
- 无其他 eQuant 子包依赖
- 被 eFactorCraft, eBacktestCraft 消费

## 安装

```bash
pip install -e eClassic-Py
```

## 快速开始

```python
from eclassic import (
    momentum, value, size, volatility, rps,
    beta, profitability, slope, return_, benchmark,
)

# 长格式面板 DataFrame
df = momentum(df, n=252)                # 252 日动量因子 (continuous/sequential)
df = value(df, bv_col="bv", cap_col="cap")  # 估值因子 B/M
df = size(df, cap_col="cap")           # 规模因子 (log 市值)
df = volatility(df, n=60)              # 波动率因子 (sd/downside)
df = rps(df, n=120)                    # 相对强度 RPS (截面排名 [0,1])
df = beta(df, n=252)                   # 滚动市场 Beta
df = slope(df, n=60)                   # 斜率因子 (alpha + beta)
df = profitability(df)                 # ROE 盈利能力
df = return_(df, n=5)                  # 收益因子
df = benchmark(df, type="excess")      # 超额收益

# 所有函数返回附加新因子列的 DataFrame
```

## API 参考

### 因子函数

所有函数统一签名：`func(df, **params) -> pd.DataFrame`。

| 函数 | 说明 | 关键参数 | 输出列 |
|------|------|---------|--------|
| `momentum(df, close_col, n, type, na_pad, new_col, append)` | 动量因子 (Jegadeesh-Titman 风格) | `n=(2,5,10)`, `type="continuous"` | `momentum_{n}` |
| `value(df, bv_col, cap_col, new_col, append)` | 价值因子 (Book-to-Market) | `bv_col="bv"`, `cap_col="cap"` | `value` |
| `size(df, cap_col, new_col, append)` | 规模因子 (log 总市值) | `cap_col="cap"` | `size` |
| `beta(df, close_col, benchmark_col, n, new_col, append)` | 贝塔因子 (滚动市场 Beta) | `n=60` | `beta_{n}` |
| `slope(df, close_col, benchmark_col, n, new_col, append)` | 斜率因子 (趋势强度) | `n=60` | `slope_alpha_{n}`, `slope_beta_{n}` |
| `volatility(df, close_col, n, type, trading_days, new_col, append)` | 波动率因子 (sd/downside) | `n=60`, `type="sd"`, `trading_days=252` | `volatility_{n}` |
| `profitability(df, op_col, bv_col, new_col, append)` | 盈利能力因子 (ROE) | `op_col="op"`, `bv_col="bv"` | `profitability` |
| `investment(df, assets_col, n, new_col, append)` | 投资因子 (总资产增长率) | `n=252`, `assets_col="assets"` | `investment_{n}` |
| `return_(df, close_col, n, type, na_pad, new_col, append)` | 收益率因子 (continuous/sequential) | `n=(1,5,21)`, `type="continuous"` | `return_{n}` |
| `sma(df, close_col, n, new_col, append)` | 均线偏离因子 | `n=(20,60,120)` | `sma_{n}` |
| `ram(df, close_col, n, risk, new_col, append)` | 风险调整动量 (RAM) | `n=252`, `risk="vol"` | `ram_{n}` |
| `rps(df, close_col, n, new_col, append)` | 相对价格强度 (截面排名) | `n=(60,120,252)` | `rps_{n}` |
| `benchmark(df, close_col, benchmark_col, type, new_col, append)` | 基准相对收益 (excess/relative) | `type="excess"` | `benchmark` |

### 通用参数说明

| 参数 | 说明 |
|------|------|
| `close_col` | 收盘价列名，默认 `"close"` |
| `bv_col` | 账面价值列名，默认 `"bv"` |
| `cap_col` | 市值列名，默认 `"cap"` |
| `assets_col` | 总资产列名，默认 `"assets"` |
| `op_col` | 营业利润列名，默认 `"op"` |
| `benchmark_col` | 基准收益率列名 |
| `new_col` | 输出列名，默认按 `{name}_{n}` 规则自动生成 |
| `append` | 是否追加到原 DataFrame，默认 `True` |
| `type` | 计算方式 (`"continuous"` / `"sequential"` / `"sd"` / `"downside"` / `"excess"` / `"relative"`) |
| `na_pad` | 是否用 NaN 填充，默认 `True` |
| `trading_days` | 年化交易日数，默认 `252` |

## 与 eTTR 的区别

| | eTTR | eClassic |
|---|---|---|
| **来源** | 技术分析 (TA) | 学术文献 (Fama-French/Barra) |
| **核心理念** | 价格形态与趋势 | 风险定价与风格溢价 |
| **典型因子** | RSI/MACD/布林带/KDJ | 动量/价值/规模/波动率/Beta |
| **计算方式** | 时序滚动窗口指标 | 截面排序 + 组合构建 |
| **数据需求** | 仅需 OHLCV | 需要财务数据 (PE/PB/市值等) |
| **应用场景** | 交易信号生成 | 风险归因 + 风格分析 |

两者互补：eTTR 提供交易信号，eClassic 提供风险归因和风格分析。

## 数据规范

所有函数接受**长格式面板 DataFrame**，返回附加新因子列的同一格式。

| date | code | close | bv | cap | assets | op | benchmark_col |
|------|------|-------|-----|------|--------|------|---------------|
| 2024-01-02 | 000001 | ... | ... | ... | ... | ... | ... |

## 与各子包的关系

- **eFactorCraft**：主要上游 — 接收 eClassic 因子进行预处理、中性化、合成、筛选
- **eBacktestCraft**：因子输入 — 可用于信号生成和权重分配
- **eTTR**：互补 — 技术与学术因子结合构成完整因子库
- **webapp**：因子选择面板中可勾选 eClassic 因子

## 版本

0.1.0 — Python >= 3.9
