# equant — 统一量化研究入口

`equant` 是 eQuant-Py 的统一聚合入口包，将所有子包整合为一个顶层命名空间，同时提供通用工具模块（面板验证、装饰器、类型别名）。

## 在 eQuant-Py 中的角色

作为**入口层**，equant 是面向用户的高级 API：

- **统一命名空间**：`from equant import ettr, eclassic, ebacktestcraft` 一键导入所有子包。
- **懒加载机制**：子包按需加载，不会在 import 时占用内存。
- **版本管理**：`equant.versions()` 查看所有子包的版本号。
- **通用工具**：`equant.utils` 提供面板验证、装饰器等共享基础设施，供各子包统一使用。

## 架构层级

```
equant (入口层)
  ├── 版本管理    → versions()
  ├── 子包懒加载  → ettr / eclassic / efactorcraft / ebacktestcraft / ealpha101 / ecandlesticks
  └── utils/      → 通用工具
       ├── validate_panel(df)         → 面板数据校验
       ├── sort_panel(df)             → 按 [date, code] 排序
       ├── ensure_columns(df, cols)   → 确保列存在
       ├── slim_output(df, cols)      → 控制输出列范围
       ├── panel_aware(func)          → 面板感知装饰器
       ├── copy_safe(func)            → 安全拷贝装饰器
       ├── with_append_output(param)  → 追加输出装饰器工厂
       └── PanelFrame                 → 类型别名 (pd.DataFrame)
```

## 依赖关系

- 依赖所有 eQuant-Py 功能子包：eTTR, eClassic, eAlpha101, eFactorCraft, eBacktestCraft, eCandleSticks
- 为所有子包提供基础设施（utils）
- 作为用户唯一的安装和导入入口

## 安装

```bash
cd eQuant-Py
pip install -e .
```

## 快速开始

```python
import equant

# 1. 查看所有子包版本
equant.versions()
# {'eQuant': '0.1.0', 'eTTR': '0.1.0', 'eClassic': '0.1.0', ...}

# 2. 通过懒加载访问子包
from equant import ettr, eclassic, ebacktestcraft

df = ettr.rsi(df, period=14)
df = eclassic.momentum(df, n=252)
result = ebacktestcraft.run(df, config)

# 3. 使用通用工具
from equant.utils import validate_panel, sort_panel, ensure_columns, slim_output

df = sort_panel(df)              # 按 [date, code] 排序
validate_panel(df)               # 校验必需的 date/code 列
df = ensure_columns(df, ["close", "volume"])  # 确保指定列存在
output = slim_output(df, "new_factor", append=False)  # 只返回 [date, code, new_factor]

# 4. 使用装饰器
from equant.utils import panel_aware, copy_safe, with_append_output

@panel_aware
def my_factor(df, period=14):
    # 输入自动 copy、validate、sort
    # 直接写业务逻辑即可
    return df

@with_append_output("new_col")
@copy_safe
def compute_something(df, n, new_col="result", append=True):
    # append 参数自动处理
    df[new_col] = df["close"].rolling(n).mean()
    return df
```

## API 参考

### utils 通用工具

| 函数 / 类 | 签名 | 说明 |
|-----------|------|------|
| `validate_panel(df, extra_cols)` | `(DataFrame, Optional[Sequence[str]]) -> None` | 验证面板必需的 `date`、`code` 列，可选验证额外列 |
| `sort_panel(df)` | `(DataFrame) -> DataFrame` | 按 `["date", "code"]` 排序 |
| `ensure_columns(df, cols)` | `(DataFrame, Sequence[str]) -> DataFrame` | 确保指定列存在，缺失时抛出异常 |
| `slim_output(df, new_col, append)` | `(DataFrame, str\|Sequence[str], bool) -> DataFrame` | `append=False` 时只返回 id 列 + 新列，适合链式计算 |
| `panel_aware(func)` | 装饰器 | 自动执行 `copy() -> validate_panel() -> sort_panel()` |
| `copy_safe(func)` | 装饰器 | 自动对输入 DataFrame 执行 `.copy()` |
| `with_append_output(new_col_param)` | 装饰器工厂 | 将 `append` 参数转换为函数返回控制 |
| `PanelFrame` | 类型别名 | `pd.DataFrame` 的语义化别名 |
| `required_id_cols()` | `() -> frozenset` | 返回 `{"date", "code"}` |

### 子包懒加载访问

```python
from equant import ettr           # → eTTR-Py
from equant import eclassic       # → eClassic-Py
from equant import ealpha101      # → eAlpha101-Py
from equant import efactorcraft   # → eFactorCraft-Py
from equant import ebacktestcraft # → eBacktestCraft-Py
from equant import ecandlesticks  # → eCandleSticks-Py
```

## 数据规范

所有子包遵循统一的**长格式面板 DataFrame**：

| date | code | open | high | low | close | volume |
|------|------|------|------|-----|-------|--------|
| 2024-01-02 | 000001 | ... | ... | ... | ... | ... |

`equant.utils` 中的所有工具函数都基于这一约定。

## 与各子包的关系

- **edatatools**：不存在入口层依赖，独立安装使用
- **eTTR / eClassic / eAlpha101**：通过 `from equant import ettr` 懒加载
- **eFactorCraft**：通过 `from equant import efactorcraft` 懒加载，同时使用 `equant.utils`
- **eBacktestCraft**：通过 `from equant import ebacktestcraft` 懒加载，同时使用 `equant.utils`
- **eCandleSticks**：通过 `from equant import ecandlesticks` 懒加载
- **eFinCharts**：独立使用，不通过 equant 导入
- **webapp**：通过 `import equant` 访问所有子包

## 版本

0.1.0 — Python >= 3.9
