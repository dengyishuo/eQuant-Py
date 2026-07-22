# eAlpha101 — WorldQuant 101 Alpha 因子

基于 Kakushadze (2016) 论文 *101 Formulaic Alphas* 的完整 Python 实现，提供 101 个量化 Alpha 因子和 17 个底层算子原语，支持长格式面板 DataFrame。

## 在 eQuant-Py 中的角色

作为**因子层**的最高级组成部分，eAlpha101 实现了业界经典的 101 个公式化 Alpha 因子：

- 这些因子由 WorldQuant 提出，是量化多因子模型的标杆
- 与 [eTTR](../eTTR-Py/)/[eClassic](../eClassic-Py/) 互补：eTTR 是技术指标，eClassic 是学术风险因子，eAlpha101 是量化公式化因子
- 输出可直接输入 [eFactorCraft](../eFactorCraft-Py/) 进行预处理/合成/筛选
- 输出可直接输入 [eBacktestCraft](../eBacktestCraft-Py/) 作为策略信号

## 架构层级

```
eAlpha101 (因子层 — 高级 Alpha)
  ├── primitives       → 底层算子原语 (17 个：ts_sum, ts_rank, correlation, decay_linear, ...)
  ├── catalog          → 因子目录 (ALPHAS, summary, required_cols, get_alpha)
  ├── data             → 示例数据加载 (load_sample_data)
  ├── formulas / gen_formulas  → 公式推导 / 生成
  ├── alpha001_020.py  → Alpha 1-20
  ├── alpha021_040.py  → Alpha 21-40
  ├── alpha041_060.py  → Alpha 41-60
  ├── alpha061_080.py  → Alpha 61-80
  └── alpha081_101.py  → Alpha 81-101
```

## 依赖关系

- 仅依赖 pandas 和 numpy
- 无其他 eQuant 子包依赖
- 被 eFactorCraft, eBacktestCraft 消费

## 安装

```bash
pip install -e eAlpha101-Py
```

## 快速开始

```python
from ealpha101 import (
    add_alpha001, add_alpha012, add_alpha046,
    summary, required_cols, get_alpha,
    load_sample_data,
)

# 1. 查看所有 Alpha 的元信息
print(summary())
#        name                description                          required_cols
# 0  alpha001  (rank(Ts_ArgMax(...)) - 0.5)  [open, close, returns, vwap, volume]
# ...

# 2. 查询特定 Alpha 需要的列
required_cols("alpha001")   # ['open', 'close', 'returns', 'vwap', 'volume']

# 3. 长格式面板 DataFrame 计算因子
df = add_alpha001(df)        # Alpha 001
df = add_alpha012(df)        # Alpha 012
df = add_alpha046(df)        # Alpha 046 (支持中性化变体)

# 4. 按名称获取 Alpha 函数
alpha_func = get_alpha("alpha032")
df = alpha_func(df)

# 5. 加载示例数据
sample = load_sample_data()  # 10 只 A 股 x 500 交易日的 OHLCV 数据

# 6. 直接使用底层原语
from ealpha101 import ts_sum, ts_rank, decay_linear, correlation, delta
result = ts_rank(df["close"], 10)
```

## API 参考

### 因子目录

| 函数 | 说明 |
|------|------|
| `load_sample_data()` | 加载 10 只 A 股 x 500 交易日的示例 OHLCV 数据 |
| `summary()` | 返回所有 Alpha 的元信息 DataFrame（序号、名称、描述、所需列） |
| `required_cols(alpha_name)` | 查询特定 Alpha 需要的输入列 |
| `get_alpha(alpha_name)` | 按名称（如 `"alpha032"`）获取 Alpha 函数 |

### Alpha 因子（101 个）

每个 `add_alphaNNN(df, **params) -> pd.DataFrame` 输入长格式面板 DataFrame，返回附加 `alpha_NNN` 列的 DataFrame。具体参数取决于各因子，由 catalog 中的元数据定义。

#### 因子分组

| Alpha 范围 | 数量 | 特点 |
|------------|------|------|
| Alpha 001-020 | 20 | 基础 Alpha，大多数需要 OHLCV + returns + vwap |
| Alpha 021-040 | 20 | 中等复杂度，部分需要行业/市值列 |
| Alpha 041-060 | 20 | 较为复杂，包含中性化和组内排名 |
| Alpha 061-080 | 20 | 高复杂度，多因子组合信号 |
| Alpha 081-101 | 21 | 最复杂，跨资产/多时间尺度 |

### 底层算子原语 (17 个)

这些底层算子构成了 Alpha 因子的计算基础，也可在自定义因子中单独使用：

| 原语 | 说明 | 示例 |
|------|------|------|
| `adv(d)` | 加权日均交易量 (Average Daily Volume) | `adv(20)` |
| `correlation(x, y, d)` | 时序滚动相关系数 | `correlation(close, returns, 20)` |
| `covariance(x, y, d)` | 时序滚动协方差 | `covariance(close, returns, 20)` |
| `cs_rank(x)` | 截面排名 (Cross-Sectional Rank) | `cs_rank(factor)` |
| `decay_linear(x, d)` | 线性衰减加权 | `decay_linear(close, 10)` |
| `delay(x, d)` | 延迟算子 | `delay(close, 1)` |
| `delta(x, d)` | 差分 | `delta(close, 5)` |
| `scale_alpha(x)` | Alpha 缩放（去均值 / 除绝对值和） | `scale_alpha(factor)` |
| `signedpower(x, a)` | 符号保持幂次 | `signedpower(returns, 0.5)` |
| `ts_argmax(x, d)` | 时序滚动 argmax | `ts_argmax(close, 20)` |
| `ts_argmin(x, d)` | 时序滚动 argmin | `ts_argmin(close, 20)` |
| `ts_max(x, d)` | 时序滚动最大值 | `ts_max(high, 20)` |
| `ts_min(x, d)` | 时序滚动最小值 | `ts_min(low, 20)` |
| `ts_product(x, d)` | 时序滚动乘积 | `ts_product(returns, 20)` |
| `ts_rank(x, d)` | 时序滚动排名 | `ts_rank(close, 252)` |
| `ts_stddev(x, d)` | 时序滚动标准差 | `ts_stddev(returns, 20)` |
| `ts_sum(x, d)` | 时序滚动求和 | `ts_sum(volume, 20)` |

## 使用案例

### 案例 1：批量计算 + 因子筛选

```python
from ealpha101 import summary, get_alpha, required_cols
from efactorcraft import ic_analysis, correlation_screen

# 筛选出只需要 OHLCV 的 Alpha
catalog = summary()
simple_alphas = [row["name"] for _, row in catalog.iterrows()
                 if all(c in df.columns for c in required_cols(row["name"]))]

# 批量计算
for name in simple_alphas[:20]:
    func = get_alpha(name)
    df = func(df)

# IC 筛选
alpha_cols = [f"alpha_{i:03d}" for i in range(1, 21)]
ic = ic_analysis(df, factor_cols=alpha_cols, forward_col="return_5d")
screened = correlation_screen(df, factor_cols=alpha_cols, threshold=0.7)
```

### 案例 2：Alpha 因子组合

```python
from ealpha101 import add_alpha001, add_alpha012, add_alpha046
from efactorcraft import equal_weighted_composite, icir_weighted_composite

# 计算选择的 Alpha
df = add_alpha001(df)
df = add_alpha012(df)
df = add_alpha046(df)

# 合成
alpha_cols = ["alpha_001", "alpha_012", "alpha_046"]
df["composite_eq"] = equal_weighted_composite(df, alpha_cols)
df["composite_icir"] = icir_weighted_composite(df, alpha_cols, forward_col="return_5d")
```

## 数据规范

- 输入：长格式面板 DataFrame
- 需要的基础列：`open`, `high`, `low`, `close`, `volume`, `returns`, `vwap`
- 部分 Alpha 需要额外列：`cap`（市值）、`industry`（行业）等
- 需要中性化的 Alpha 接受以 `_neut` 结尾的预计算列
- 输出：附加 `alpha_NNN` 列的 DataFrame

## 参考文献

Kakushadze, Z. (2016). *101 Formulaic Alphas*. Wilmott, 2016(84), 72-81.

## 与各子包的关系

- **eFactorCraft**：主要下游 — 接收 eAlpha101 因子进行预处理/合成/筛选/择时
- **eBacktestCraft**：因子输入 — Alpha 因子可作为 `signal()` 的输入
- **eTTR / eClassic**：互补 — 三类因子（技术/学术/量化）覆盖完整的因子谱系
- **webapp**：因子选择面板中可勾选 Alpha 因子

## 版本

0.1.0 — Python >= 3.9
