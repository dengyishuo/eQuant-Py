# webapp — 量化回测工作台

基于 Streamlit 的可视化回测 Web 应用，将 eQuant-Py 全工具链整合为三页式交互工作流，覆盖从数据配置、因子计算到信号生成和回测评估的完整闭环。

## 在 eQuant-Py 中的角色

作为**应用层**，webapp 是整个工具链的最终呈现平台。它将各子包串联为一个完整的可视化工作台：

- [edatatools](../edatatools/) — 数据基础设施（交易日历、股票池）
- [eTTR](../eTTR-Py/) / [eClassic](../eClassic-Py/) / [eAlpha101](../eAlpha101-Py/) — 因子计算引擎
- [eFactorCraft](../eFactorCraft-Py/) — 因子工程流水线
- [eCandleSticks](../eCandleSticks-Py/) — 形态信号识别
- [eBacktestCraft](../eBacktestCraft-Py/) — 回测引擎
- [eFinCharts](../eFinCharts-Py/) — 行情图表可视化

用户可以在浏览器中完成从数据拉取到策略回测的全流程，无需编写代码。

## 架构层级

```
webapp (应用层)
  ├── app.py                          → 主页入口 (策略列表 / 加载 / 创建)
  ├── pages/
  │   ├── 1_数据源配置.py              → 数据源选择 + 连接测试
  │   ├── 2_股票池与因子.py            → 行情拉取 + 因子计算
  │   └── 3_信号与回测.py              → 信号生成 + 回测运行 + 结果可视化 + 策略保存
  └── lib/                             → 基础设施
       ├── bootstrap.py                 → sys.path 修正，确保子包可导入
       ├── config_store.py              → 策略持久化 (JSON)
       ├── data_cache.py                → 数据本地缓存 (Parquet)
       └── factor_catalog.py            → 因子目录构建 (自省 eclassic/ettr/ealpha101)
```

## 依赖关系

整合 eQuant-Py **全部**功能子包，外部依赖：streamlit, pandas, matplotlib, yfinance, akshare, tushare, baostock。

## 安装与启动

```bash
cd webapp
pip install -r requirements.txt

# 启动 (必须在 eQuant-Py 根目录下，确保子包可导入)
cd ..
streamlit run webapp/app.py
```

## 页面功能详解

### 主页 — 策略管理

- 展示已保存的策略列表
- 加载已有策略继续编辑
- 创建新策略
- 删除/重命名已有策略

```python
# lib/config_store.py 提供的 API
from webapp.lib.config_store import (
    load_strategy, save_strategy, delete_strategy, list_strategies,
    load_config, save_config,
)
```

### 页面 1 — 数据源配置

- 选择数据源：Yahoo Finance / Tushare / AKShare / baostock
- 配置对应 token 或参数
- 测试连接状态
- 设置数据缓存策略

### 页面 2 — 股票池与因子

- 录入或导入股票代码列表
- 拉取历史行情数据 (OHLCV)
- 勾选并批量计算因子，覆盖三大因子库和因子工程：
  - **eTTR**: RSI, MACD, ATR, ADX, Bollinger, Stoch, KDJ, OBV 等 55+ 个技术指标
  - **eClassic**: momentum, value, size, volatility, beta, rps 等经典因子
  - **eAlpha101**: WorldQuant 101 Alpha 因子
  - **eFactorCraft**: 标准化、行业中性化、市值中性化、多因子合成、因子筛选
- 因子参数可自定义（周期、平滑等）
- 预览因子计算结果

```python
# lib/factor_catalog.py 提供的 API
from webapp.lib.factor_catalog import (
    available_factors, build_catalog,
    FactorSpec,  # 因子元数据: name, category, function, params
    ParamSpec,   # 参数元数据: name, type, default, min, max
)
```

### 页面 3 — 信号与回测

- 配置交易信号生成规则：
  - 25 种信号类型可选 (quantile, threshold, crossover, ma_cross, ...)
  - Top-N 选股参数
  - 多条件组合 (AND/OR/VOTE)
- 设置权重方案 (等权/因子比例/波动率平价/风险平价 等)
- 配置回测参数：
  - 初始资金、交易费率、印花税、滑点
  - 止盈止损规则（固定/移动/ATR/波动率/OCO 模式）
- 运行回测
- 查看绩效图表：
  - 净值曲线 + 回撤图
  - 月度收益热力图
  - 收益率分布图
  - 基准对比图
- 导出/保存策略

## 数据流

```
数据源配置 → 股票池选择 → 行情拉取 → 因子计算 → 信号生成 → 回测运行 → 结果展示 + 保存
    │            │            │          │          │          │
    v            v            v          v          v          v
 Tushare     [000001,    get_data()  eTTR/     signal()    run()     plot_all()
 AKShare      000002,    (eFactor    eClassic/  (eBacktest  (eBacktest
 baostock     ...]        Craft)     eAlpha101/  Craft)      Craft)
 Yahoo                              eFactorCraft
```

## lib 层公共 API

| 模块 | 类 / 函数 | 说明 |
|------|-----------|------|
| `bootstrap` | 自动执行 | sys.path 修正，确保 `import equant` 可用 |
| `config_store` | `load_config()` / `save_config(cfg)` | `~/.equant/config.json` 读写 |
| `config_store` | `list_strategies()` / `load_strategy(name)` / `save_strategy(name, data)` / `delete_strategy(name)` | 策略 JSON 持久化在 `~/.equant/strategies/` |
| `data_cache` | `get_cached(source, universe, start_date, end_date)` / `set_cached(...)` | 数据本地 Parquet 缓存 (`~/.equant/cache/`) |
| `factor_catalog` | `available_factors(columns)` / `build_catalog()` | 自省 eclassic/ettr/ealpha101 构建 UI 可用的因子列表 |

### FactorSpec 和 ParamSpec

```python
@dataclass
class ParamSpec:
    name: str           # 参数名
    type: str           # int / float / str
    default: Any        # 默认值
    min: Optional[Any]  # 最小值 (numeric 类型)
    max: Optional[Any]  # 最大值 (numeric 类型)

@dataclass
class FactorSpec:
    name: str           # 因子名称 (如 "rsi")
    category: str       # 分类 (ettr / eclassic / ealpha101)
    description: str    # 描述
    function: Callable  # 因子函数
    params: List[ParamSpec]  # 参数列表
    required_cols: List[str]  # 需要的列
```

## 技术栈

- **前端**: Streamlit
- **数据**: pandas, numpy, pyarrow (Parquet 缓存)
- **图表**: matplotlib, seaborn
- **数据源**: yfinance, akshare, tushare, baostock
- **持久化**: JSON (配置/策略), Parquet (行情缓存)

## 扩展指南

添加新的因子到 UI 目录中：

```python
# 在 lib/factor_catalog.py 的 build_catalog() 中添加
from ettr import my_new_indicator

catalog.append(FactorSpec(
    name="my_new_indicator",
    category="ettr",
    description="My custom indicator",
    function=my_new_indicator,
    params=[ParamSpec("period", "int", 14, 2, 252)],
    required_cols=["close"],
))
```

## 数据存储路径

```
~/.equant/
  ├── config.json          → 全局配置
  ├── cache/               → 行情数据缓存 (Parquet)
  └── strategies/          → 策略 JSON
       ├── strategy_1.json
       └── strategy_2.json
```

## 与各子包的关系

webapp 作为最顶层的应用层，集成 eQuant-Py 的全部子包：

| 子包 | 在 webapp 中的用途 | 对应页面 |
|------|-------------------|---------|
| edatatools | 交易日历、股票池构建 | 页面 2 |
| eTTR | 技术指标因子计算 | 页面 2 |
| eClassic | 经典因子计算 | 页面 2 |
| eAlpha101 | Alpha 因子计算 | 页面 2 |
| eFactorCraft | 数据获取、因子预处理/合成/筛选 | 页面 2 |
| eCandleSticks | 形态信号输入 | 页面 3 |
| eBacktestCraft | 信号生成、权重分配、回测引擎、绩效分析、图表 | 页面 3 |
| eFinCharts | K 线图展示 | 页面 3 |
| equant | 统一导入入口 + 面板工具 | 全局 |

## 版本

0.1.0 — Python >= 3.9
