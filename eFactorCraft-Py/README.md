# eFactorCraft — 因子工程流水线

从数据采集、预处理、中性化、IC/IR 分析、多因子合成到因子筛选与择时的完整因子工程工具包。它是因子研发生命周期的管理中心。

## 在 eQuant-Py 中的角色

作为**因子工程层**，eFactorCraft 扮演因子管理中心角色：

- 它**不产生**新因子，而是对来自 [eTTR](../eTTR-Py/)、[eClassic](../eClassic-Py/)、[eAlpha101](../eAlpha101-Py/) 的原始因子进行精炼和筛选
- 提供完整的因子工程流水线：预处理 → 分析 → 合成 → 筛选 → 择时
- 支持多数据源获取（yahoo/tushare/akshare/baostock）
- 最终输出高质量的合成因子，供 [eBacktestCraft](../eBacktestCraft-Py/) 回测使用
- 上游依赖因子层，下游供给策略层

## 架构层级

```
eTTR / eClassic / eAlpha101 (因子层)
              │
              ▼
eFactorCraft (因子工程层)
  ├── 数据获取 (data.py)       → get_data (多源数据采集)
  ├── 数据供应商 (providers/)   → akshare / baostock / tushare / yfinance
  ├── 预处理 (preprocess.py)   → winsorize, standardize, industry_neutralize, size_neutralize, factor_preprocess
  ├── 分析 (analysis.py)       → add_next_return, ic_analysis, ir_analysis, quantile_analysis
  ├── 排名 (rank.py)           → quantile_rank, quantile_flag, consecutive_days
  ├── 因子合成 (synthesis/)     → 6 种复合方法 (等权/IC/ICIR/衰减/PCA/排名)
  ├── 因子筛选 (selection/)     → 5 种筛选方法 (相关性/IC/稳定性/Top-N/综合报告)
  ├── 因子择时 (timing/)        → 5 种择时方法 (自适应/市场状态/择时权重/趋势/波动率)
              │
              ▼
eBacktestCraft (策略层)
```

## 依赖关系

- pandas, numpy, scipy, statsmodels, requests, yfinance
- 上游依赖 eTTR / eClassic / eAlpha101 等因子层（数据流入）
- 下游供给 eBacktestCraft（合成因子输出）
- 可选数据供应商：akshare, baostock, tushare

## 安装

```bash
pip install -e eFactorCraft-Py
```

## 快速开始

```python
from efactorcraft import (
    get_data,
    winsorize, standardize, industry_neutralize, size_neutralize, factor_preprocess,
    add_next_return, ic_analysis, ir_analysis, quantile_analysis,
    quantile_rank, quantile_flag, consecutive_days,
    equal_weighted_composite, icir_weighted_composite, correlation_screen, factor_report,
)

# 1. 获取数据
df = get_data(codes=["000001", "000002"], start="20240101", end="20241231")

# 2. 附加前瞻收益（作为标签）
df = add_next_return(df, periods=(1, 5, 20))

# 3. 预处理 — 缩尾 + 标准化 + 行业中性化 + 市值中性化
df = winsorize(df, factor_col=["factor1", "factor2"], probs=(0.01, 0.99))
df = standardize(df, factor_col=["factor1", "factor2"])
df = industry_neutralize(df, factor_col=["factor1", "factor2"], industry_col="industry")
df = size_neutralize(df, factor_col=["factor1", "factor2"], size_col="cap")

# 4. IC 分析
ic = ic_analysis(df, factor_cols=["factor1", "factor2"], forward_col="return_1d", method="spearman")

# 5. 多因子合成
df["composite"] = icir_weighted_composite(df, factor_cols=["factor1", "factor2"], forward_col="return_1d")

# 6. 因子筛选
screened = correlation_screen(df, factor_cols=["factor1", "factor2"], threshold=0.7)
report = factor_report(df, factor_cols=["factor1", "factor2"], forward_col="return_1d")
```

## API 参考

### 预处理 (Preprocess)

| 函数 | 说明 | 关键参数 |
|------|------|---------|
| `winsorize(df, factor_col, probs, by, new_col_prefix, append)` | 截面缩尾处理 | `probs=(0.01, 0.99)`, `by="date"` |
| `standardize(df, factor_col, by, new_col_prefix, append)` | 截面 z-score 标准化 | `by="date"` |
| `industry_neutralize(df, factor_col, industry_col, by, new_col_prefix, min_samples, append)` | OLS 行业中性化 | `industry_col="industry"`, `min_samples=5` |
| `size_neutralize(df, factor_col, size_col, by, new_col_prefix, min_samples, append)` | OLS 市值中性化 | `size_col="cap"`, `min_samples=5` |
| `factor_preprocess(df, factor_col, industry_col, size_col, probs, by, append)` | 一键预处理流水线 | 依次执行 winsorize → standardize → industry → size |

### 分析 (Analysis)

| 函数 | 说明 | 关键参数 |
|------|------|---------|
| `add_next_return(df, close_col, periods, new_col, append)` | 附加未来 N 期收益（作为预测标签） | `periods=(1,5,20)` |
| `ic_analysis(df, factor_cols, forward_col, method)` | IC/Rank IC 分析，返回 `dict` | `method="spearman"` / `"pearson"` |
| `ir_analysis(ic_results)` | 基于 IC 结果计算 Information Ratio | 传入 `ic_analysis` 结果 |
| `quantile_analysis(df, factor_col, forward_col, n_groups, by)` | 分位数组合分析 | `n_groups=10` |

### 排名 (Rank)

| 函数 | 说明 | 关键参数 |
|------|------|---------|
| `quantile_rank(df, factor_cols, type, n, by, ties_method, new_col, append)` | 分位数排名 | `type="cross"` / `"time"`, `n=60` |
| `quantile_flag(df, factor_col, n_groups, by, new_col, append)` | 分组标签 | `n_groups=10` |
| `consecutive_days(df, flag_col, new_col, append)` | 连续满足条件的天数 | — |

### 数据 (Data)

| 函数 | 说明 | 关键参数 |
|------|------|---------|
| `get_data(codes, start_date, end_date, source, progress)` | 多数据源统一数据获取 | `source="yahoo"` / `"tushare"` / `"akshare"` / `"baostock"` |

返回长格式面板 DataFrame（date + code + OHLCV）。

### 因子合成 (Synthesis)

| 函数 | 说明 |
|------|------|
| `equal_weighted_composite(df, factor_cols, ...)` | 等权合成 |
| `ic_weighted_composite(df, factor_cols, forward_col, ...)` | IC 加权合成 |
| `icir_weighted_composite(df, factor_cols, forward_col, ...)` | ICIR 加权合成 |
| `max_decay_composite(df, factor_cols, forward_col, ...)` | 最大衰减加权 |
| `pca_composite(df, factor_cols, ...)` | PCA 主成分合成 |
| `rank_weighted_composite(df, factor_cols, forward_col, ...)` | 排名加权合成 |

### 因子筛选 (Selection)

| 函数 | 说明 |
|------|------|
| `correlation_screen(df, factor_cols, threshold)` | 去重筛选（保留与收益相关性最高的去相关因子） |
| `ic_screen(df, factor_cols, forward_col, threshold)` | IC 阈值筛选 |
| `stability_screen(df, factor_cols, ...)` | 稳定性筛选 |
| `select_top(df, factor_cols, forward_col, n)` | Top-N 选择 |
| `factor_report(df, factor_cols, forward_col)` | 综合因子报告（IC/IR/相关性/稳定性） |

### 因子择时 (Timing)

| 函数 | 说明 |
|------|------|
| `adaptive_composite(df, factor_cols, ...)` | 自适应合成权重 |
| `regime_detect(df, ...)` | 市场状态检测（趋势/震荡/高波动） |
| `timing_weight(df, factor_cols, ...)` | 择时权重分配 |
| `trend_filter(df, ...)` | 趋势过滤（仅在趋势市中启用因子） |
| `vol_filter(df, ...)` | 波动率过滤（高波动时降低权重） |

### 数据供应商 (Providers)

| 模块 | 说明 |
|------|------|
| `providers.set_token(source, token)` | 设置数据源 token |
| `providers.tushare.get_daily(code, start_date, end_date)` | Tushare 日线数据 |
| `providers.akshare.get_daily(code, start_date, end_date)` | AKShare 日线数据 |
| `providers.baostock.get_daily(code, start_date, end_date)` | baostock 日线数据 |

## 使用案例

### 案例 1：一键预处理流水线

```python
from efactorcraft import factor_preprocess, ic_analysis, factor_report

# 原始因子列表
factor_cols = ["eTTR_rsi_14", "eTTR_macd", "eClassic_momentum_252", "alpha_001"]

# 一键预处理
df = factor_preprocess(df,
    factor_col=factor_cols,
    industry_col="industry",
    size_col="cap",
    probs=(0.01, 0.99),
)

# 分析 + 报告
ic = ic_analysis(df, factor_cols=factor_cols, forward_col="return_5d", method="spearman")
report = factor_report(df, factor_cols=factor_cols, forward_col="return_5d")
```

### 案例 2：完整因子工程流水线

```python
from efactorcraft import (
    factor_preprocess, ic_analysis, correlation_screen,
    icir_weighted_composite, factor_report, get_data,
)
from ettr import rsi, macd
from eclassic import momentum

# 1. 获取数据
df = get_data(codes=["000001", "000002", "000003"], start="20240101", end="20241231")

# 2. 计算原始因子（来自因子层）
df = rsi(df, period=14)
df = macd(df, fast=12, slow=26, signal=9)
df = momentum(df, n=252)

# 3. 预处理
factor_cols = ["rsi_14", "macd_12_26_9", "momentum_252"]
df = factor_preprocess(df, factor_col=factor_cols,
                       industry_col="industry", size_col="cap",
                       probs=(0.01, 0.99))

# 4. IC 分析
ic = ic_analysis(df, factor_cols=factor_cols, forward_col="return_5d", method="spearman")

# 5. 去重筛选
selected = correlation_screen(df, factor_cols=factor_cols, threshold=0.7)

# 6. ICIR 加权合成
df["composite"] = icir_weighted_composite(df, selected, forward_col="return_5d")

# 7. 综合报告
report = factor_report(df, selected + ["composite"], forward_col="return_5d")
```

## 数据规范

所有函数接受**长格式面板 DataFrame**：

| date | code | close | factor1 | factor2 | industry | cap | return_1d | ... |
|------|------|-------|---------|---------|----------|------|-----------|-----|

返回附加处理结果的同一格式。中性化/标准化后的列名规则：`{原列名}_{处理后缀}`。

## 与各子包的关系

- **eTTR / eClassic / eAlpha101**（上游）：接收原始因子 → 预处理 → 分析 → 合成
- **eBacktestCraft**（下游）：输出合成因子 → `run()` 回测
- **edatatools**：可选依赖，用于交易日历中的日期范围查询
- **webapp**：应用层核心依赖，提供数据获取和因子工程的全部能力

## 版本

0.1.0 — Python >= 3.9
