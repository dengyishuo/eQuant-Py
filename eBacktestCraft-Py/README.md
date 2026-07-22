# eBacktestCraft — 专业量化回测框架

事件驱动的多资产策略回测引擎，支持 25+ 种信号生成、9+ 种权重方案、完整的止盈止损体系、18 项绩效指标、参数扫描和策略增强模块。

## 在 eQuant-Py 中的角色

作为**策略层**，eBacktestCraft 是将因子转化为可量化评估策略的核心环节：

- 接收来自 [eTTR](../eTTR-Py/)/[eClassic](../eClassic-Py/)/[eAlpha101](../eAlpha101-Py/) 的因子信号
- 接收 [eFactorCraft](../eFactorCraft-Py/) 输出的合成因子
- 执行完整的事件驱动回测流程
- 生成全套绩效报告和可视化图表
- 作为 [webapp](../webapp/) 的后端引擎

## 架构层级

```
因子层 (eTTR/eClassic/eAlpha101) + 因子工程 (eFactorCraft)
                      │
                      ▼
eBacktestCraft (策略层)
  ├── 核心引擎 (engine.py)        → Config + BacktestResult + run()
  ├── 信号生成 (signals.py)       → signal() — 25 种信号类型
  ├── 权重分配 (weights.py)       → equal_weight / fixed_weight / norm_weight / weight()
  ├── 绩效分析 (analytics.py)     → performance_analysis() — 18 项指标
  ├── 基准对比 (benchmark.py)     → 等权/B&H/指数基准 + 多策略对比
  ├── 参数扫描 (param_scan.py)    → param_grid / run_param_scan / best_params
  ├── 可视化 (plot.py)            → 净值/回撤/月度收益/分布图/基准对比
  └── 策略增强 (enhance/)          → 高级信号/权重/风控
       ├── signals.py             → quantile / persistent / smoothed 信号
       ├── weights.py             → vol_parity / target_vol / erp / confidence 权重
       └── risk.py                → apply_vol_target / compute_turnover
```

## 依赖关系

- pandas, numpy, matplotlib, seaborn
- 上游依赖因子层 (eTTR/eClassic/eAlpha101/eFactorCraft) — 不强制导入
- 可视化部分复用 matplotlib/seaborn

## 安装

```bash
pip install -e eBacktestCraft-Py
```

## 快速开始

```python
from ebacktestcraft import Config, run, signal, equal_weight, add_indicator, plot_all, performance_analysis

# 1. 配置回测
config = Config(
    start_date="2024-01-01",
    end_date="2024-12-31",
    init_capital=1_000_000,
    commission=0.001,
    slippage=0.001,
)

# 2. 计算因子（统一路由）
df = add_indicator(df, "momentum", n=20)

# 3. 生成信号
signals = signal(df, indicator_cols=["mom_20"], signal_type="quantile", top_n=10)

# 4. 分配权重
weights = equal_weight(signals, signal_col="composite_signal")

# 5. 运行回测
result = run(df, config, weight_col=weights.columns[-1])

# 6. 查看绩效
print(result.summary())
# {'total_return': 0.15, 'annual_return': 0.15, 'sharpe_ratio': 1.2,
#  'max_drawdown': -0.12, 'win_rate': 0.55, 'calmar_ratio': 1.25}

# 7. 一键出图
plot_all(result, save_dir="charts/", show=True)

# 8. 对比基准
from ebacktestcraft import buy_and_hold_benchmark, compare_benchmarks
bm = buy_and_hold_benchmark(df, config)
comparison = compare_benchmarks([result, bm], labels=["Strategy", "B&H"])
```

## API 参考

### 核心引擎

| 类 / 函数 | 说明 |
|-----------|------|
| `Config(start_date, end_date, init_capital, ...)` | 回测配置数据类，50+ 参数：资金、费率、印花税、滑点、手数、持仓限制、止盈止损 |
| `BacktestResult` | 回测结果对象：`daily_positions`, `equity_curve`, `transactions`, `config`, `summary` |
| `run(df, config, weight_col, **kwargs)` | 事件驱动多资产回测引擎，`df` 为长格式面板 DataFrame + OHLCV |

**Config 核心参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `start_date / end_date` | 回测起止日期 | — |
| `init_capital` | 初始资金 | `1_000_000` |
| `fee_rate` | 交易费率 | `0.0003` |
| `stamp_tax` | 印花税率 | `0.001` |
| `slippage_rate` | 滑点率 | `0.001` |
| `lot_size` | 每手股数 | `100` |
| `single_max_weight` | 单标的权重上限 | `0.1` |
| `global_max_hold_pct` | 全局仓位上限 | `1.0` |
| 止盈止损参数 | `profit_stop_*`, `loss_stop_*`, OCO 模式等 | — |
| `rebalance_mode` | 调仓模式 | `"daily"` |
| `rebalance_cycle` | 调仓周期 | `1` |

### 统一指标路由

`add_indicator(df, indicator, **kwargs) -> pd.DataFrame`

一键计算任何因子/指标，自动路由到对应的子包：

```python
add_indicator(df, "rsi", close_col="close", n=14)              # →    ettr.rsi()
add_indicator(df, "momentum", n=20)                             # → eclassic.momentum()
add_indicator(df, "alpha001", close_col="close")                # → ealpha101.add_alpha001()
add_indicator(df, "doji")                                       # → ecandlesticks.add_doji()
add_indicator(df, "eClassic.volatility", close_col="close", n=20)  # 消歧义
```

`list_indicators(package=None) -> DataFrame`

列出所有可用指标（220+ 个），可按包过滤：

```python
list_indicators()           # 全部 220+ 行 DataFrame
list_indicators("ettr")     # 仅 eTTR 的 50+ 个指标
list_indicators("eclassic") # 仅 eClassic 的 13 个因子
```

**路由表概览**：

| 包 | 指标数 | 短名示例 | 实际调用 |
|----|--------|---------|----------|
| eTTR | 50+ | `rsi`, `sma`, `macd`, `atr`, ... | `ettr.rsi()` |
| eClassic | 13 | `momentum`, `beta`, `size`, ... | `eclassic.momentum()` |
| eAlpha101 | 101 | `alpha001`–`alpha101` 或 `001`–`101` | `ealpha101.add_alpha001()` |
| eCandleSticks | 52 | `doji`, `hammer`, `engulfing`, ... | `ecandlesticks.add_doji()` |

名称冲突用 `"包名.指标"` 消歧义：`"eClassic.sma"` / `"eTTR.sma"`。

### 信号生成

`signal(df, indicator_cols, signal_type, **kwargs) -> pd.DataFrame`

支持 **25 种**信号类型：

| 类别 | 信号类型 | 说明 |
|------|---------|------|
| 阈值类 | `threshold`, `between`, `crossover` | 阈值触发 / 区间 / 交叉 |
| 排名类 | `rank`, `quantile`, `percentile`, `zscore`, `score` | 截面排名信号 |
| 滚动类 | `rolling`, `consecutive`, `window` | 滚动窗口信号 |
| 技术类 | `ma_cross`, `breakout`, `mean_reversion` | 均线/突破/回归 |
| 状态类 | `regime`, `vol_regime`, `td_setup` | 市场状态信号 |
| 逻辑类 | `and`, `or`, `vote`, `multi_condition` | 多条件组合 |
| 事件类 | `earnings`, `index_rebalance`, `macro` | 事件驱动 |
| 常数类 | `constant` | 恒定为 1 |

### 权重分配

| 函数 | 说明 |
|------|------|
| `equal_weight(df, signal_col, weight_name, zero_na)` | 1/n 等权 |
| `fixed_weight(df, signal_col, fixed_weights)` | 固定权重 |
| `norm_weight(df, weight_col, signal_col, norm_method)` | 因子比例权重 (linear/softmax) |
| `weight(df, weight_type, **kwargs)` | 统一调度器，支持 9+ 种权重：equal/fixed/norm/rank/inv_vol/target_vol/min_var/min_es/min_mdd/max_calmar/max_treynor/risk_parity/max_sharpe |

### 绩效分析

`performance_analysis(df, transactions, init_capital, risk_free_rate, periods_per_year) -> dict`

返回 18 项绩效指标，包括：总收益、年化收益、年化波动率、夏普比率、最大回撤、Calmar 比率、胜率、盈亏比、换手率等。

### 基准对比

| 函数 | 说明 |
|------|------|
| `equal_weight_benchmark(df, config)` | 等权基准 |
| `buy_and_hold_benchmark(df, config)` | 买入持有基准 |
| `index_benchmark(benchmark_df, config)` | 指数基准 |
| `compare_benchmarks(results, labels)` | 多策略/基准对比，返回对比 DataFrame |

### 可视化

| 函数 | 说明 |
|------|------|
| `plot_equity_curve(result)` | 净值曲线 |
| `plot_drawdown(result)` | 回撤曲线 |
| `plot_return_drawdown(result)` | 净值 + 回撤组合图 |
| `plot_return_dist(result)` | 收益率分布 (直方图 + KDE) |
| `plot_monthly_return(result)` | 月度收益热力图 |
| `plot_benchmark_compare(results)` | 多策略净值对比 |
| `plot_all(result, save_dir, show)` | 一键生成全部标准图表 |
| `theme_quant()` | 量化主题配色 |

### 参数扫描

| 函数 | 说明 |
|------|------|
| `param_grid(**kwargs)` | 生成参数网格 |
| `run_param_scan(df, grid, ...)` | 运行参数扫描 |
| `rank_param_scan(...)` | 排名扫描结果 |
| `sensitivity_table(results)` | 敏感性分析表 |
| `best_params(results, metric)` | 最佳参数 (按指定指标) |

### 策略增强 (enhance)

| 函数 | 模块 | 说明 |
|------|------|------|
| `quantile_signal(signals, ...)` | enhance.signals | 分位数信号生成 |
| `persistent_signal(signals, ...)` | enhance.signals | 持续性信号 (降低换手) |
| `smoothed_signal(signals, ...)` | enhance.signals | 平滑信号 |
| `vol_parity_weight(signals, ...)` | enhance.weights | 波动率平价权重 |
| `target_vol_weight(signals, target, ...)` | enhance.weights | 目标波动率权重 |
| `erp_weight(signals, ...)` | enhance.weights | 风险平价权重 |
| `confidence_weight(signals, ...)` | enhance.weights | 置信度加权 |
| `apply_vol_target(weights, ...)` | enhance.risk | 波动率目标调整 |
| `compute_turnover(weights)` | enhance.risk | 换手率计算 |

## 使用案例

### 案例 1：完整信号 → 权重 → 回测流水线

```python
from ebacktestcraft import Config, run, signal, equal_weight, plot_all, performance_analysis

config = Config(start_date="2024-01-01", end_date="2024-12-31", init_capital=1_000_000)

# 基于合成因子生成 top-10 信号
signals = signal(df, indicator_cols=["composite"], signal_type="quantile", top_n=10)
weights = equal_weight(signals, signal_col="composite_signal")

# 回测
result = run(df, config, weight_col=weights.columns[-1])

# 结果
metrics = performance_analysis(result.equity_curve, result.transactions, config.init_capital)
plot_all(result, save_dir="charts/")
```

### 案例 2：参数扫描 → 最佳参数

```python
from ebacktestcraft import Config, run, signal, equal_weight, param_grid, run_param_scan, best_params

config = Config(start_date="2024-01-01", end_date="2024-12-31", init_capital=1_000_000)

# 生成参数网格
grid = param_grid(
    indicator_cols=[["rsi_14"], ["macd_12_26_9"], ["composite"]],
    signal_type=["quantile", "threshold"],
    top_n=[5, 10, 20],
)

# 运行扫描
results = run_param_scan(df, grid, config=config)

# 按夏普找最佳参数
best = best_params(results, metric="sharpe_ratio")
```

### 案例 3：完整止损止盈回测

```python
config = Config(
    start_date="2024-01-01", end_date="2024-12-31",
    init_capital=1_000_000,
    # 止盈
    profit_stop_type="fixed", profit_stop_value=0.3,
    # 止损
    loss_stop_type="trailing_atr", loss_stop_value=2.0, loss_stop_atr_period=14,
    # 或 OCO 模式：止盈止损同时触发
    oco_mode=True,
)

signals = signal(df, indicator_cols=["composite"], signal_type="quantile", top_n=10)
weights = equal_weight(signals, signal_col="composite_signal")
result = run(df, config, weight_col=weights.columns[-1])
```

## 数据规范

- 输入：长格式面板 DataFrame，需含 `date`、`code` + OHLCV + 因子列
- Config：通过 `Config` 数据类集中管理所有回测参数
- 结果：`BacktestResult` 对象封装净值序列、持仓记录、交易记录、绩效指标

## 与各子包的关系

- **eTTR / eClassic / eAlpha101**（上游因子）：接收因子信号作为 `signal()` 的输入
- **eFactorCraft**（上游因子工程）：接收合成因子 → 回测评估
- **eFinCharts**：职责互补 — eFinCharts 负责行情图表，eBacktestCraft 负责策略绩效图表
- **webapp**（下游应用）：整个回测引擎的后端，驱动 webapp 的回测页面

## 版本

0.1.0 — Python >= 3.9
