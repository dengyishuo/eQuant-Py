# edatatools — 中国市场数据基础设施

中国 A 股量化研究的基础数据工具包，提供交易日历、CACS 复权累计收益计算和股票池构建三大核心能力。

## 在 eQuant-Py 中的角色

作为**数据层**，edatatools 是整个 eQuant-Py 的数据基础。它为所有上层模块（因子计算、回测引擎、图表可视化）提供：

- **交易日历**：标准化的 A 股交易日/非交易日映射，日期范围生成，月份偏移
- **CACS 累计收益**：正确处理 A 股除权除息的复权收益计算
- **股票池构建**：基于流动性、指数成分、固定列表的股票筛选

它是 eQuant-Py 数据流的起点，所有需要按交易日对齐的计算都依赖它。

## 架构层级

```
因子层 / 策略层 (eTTR, eFactorCraft, eBacktestCraft, ...)
        │
        ▼
  edatatools (数据层)
  ├── calendars  → 交易日历 (TradingCalendar, date_to_bus_date, date_range, ...)
  ├── returns    → CACS 复权累计收益 (cal_cum_ret)
  └── universe   → 股票池构建 (build_universe)
```

## 依赖关系

- 仅依赖 pandas 和 numpy
- 无其他 eQuant 子包依赖（最底层）
- 可选外部数据源：akshare 或 tushare（用于交易日历懒加载）
- 被所有上层模块依赖

## 安装

```bash
pip install -e edatatools
```

## 快速开始

```python
from edatatools import (
    cn_calendar, TradingCalendar,
    date_to_bus_date, date_is_bus_date, date_bus_diff, date_range,
    register_calendar,
    cal_cum_ret,
    build_universe,
)

# ---- 交易日历 ----

# 使用内置 A 股日历 (惰性加载，依次尝试 akshare → tushare)
date_to_bus_date("2024-01-06")              # 2024-01-05 (周五，最近交易日)
date_is_bus_date("2024-01-08")              # True
date_bus_diff("2024-01-02", "2024-01-31")   # 21

# 日期范围
date_range("2024-01-01", "2024-01-31")      # 1月所有交易日
date_range("2024-01-01", n=10)              # 从1/1起10个交易日

# 自定义日历
my_cal = TradingCalendar("CN", dates=my_dates)
register_calendar("CN", my_cal)

# ---- CACS 累计收益 ----

ret = cal_cum_ret(
    jtids=["000001", "000002"],
    start_dates="2024-01-01",
    end_dates="2024-12-31",
    close=close_df,           # index=date, columns=jtid
    adj_factor=adj_df,        # 可选，含 adjustingFactor / adjustingConst
)
# 返回 np.ndarray[shape=(n_stocks,)], NaN 表示数据缺失

# ---- 股票池构建 ----

# 固定列表
univ = build_universe("2024-01-01", "2024-06-30",
                      univ_type="fixed",
                      fixed_list=["000001", "000002"])

# 流动性筛选
univ = build_universe("2024-01-01", "2024-06-30",
                      univ_type="liquidity_based",
                      univ_range=(0.3, 1.0),
                      liquidity_data=liquidity_df)

# 沪深300成分
univ = build_universe("2024-01-01", "2024-06-30",
                      univ_type="CSI300",
                      index_constituents=constituents_dict)

# 返回 dict[str, list[str]] — {date_str: [jtid, ...]}
```

## API 参考

### calendars — 交易日历

核心类 `TradingCalendar` 维护一个有序的交易日列表，通过最近邻插值将任意日期映射到最近的交易日。

| 类 / 函数 | 说明 |
|-----------|------|
| `TradingCalendar(region, dates=None)` | 创建日历对象，惰性加载内置数据 |
| `cn_calendar` | 预置的 A 股日历单例 |
| `cal.to_bus_date(d, shift=0, forward=True)` | 日期 → 最近交易日 |
| `cal.is_bus_date(d)` | 判断是否为交易日 |
| `cal.bus_diff(from_date, to_date)` | 两交易日之间的天数 |
| `cal.date_range(start, end=None, n=None)` | 生成交易日范围 |
| `cal.shift_months(months, shift)` | YYYYMM 月份偏移，返回 YYYYMM |
| `cal.first_bus_date_of_month(d)` | 当月第一个交易日 |
| `cal.last_bus_date_of_month(d)` | 当月最后一个交易日 |
| `cal.from_tushare(region, token)` | 从 Tushare trade_cal 创建日历 |
| `date_to_bus_date(d, region, shift, forward)` | 模块级便捷函数 |
| `date_is_bus_date(d, region)` | 模块级便捷函数 |
| `date_bus_diff(from_date, to_date, region)` | 模块级便捷函数 |
| `date_range(start_date, end_date, n, region)` | 模块级便捷函数 |
| `register_calendar(region, cal)` | 注册/覆盖区域日历 |

**关键设计**：

- 惰性加载：首次访问时尝试 akshare（免费、无需 token），失败后尝试 tushare
- 最近邻插值：非交易日自动映射到最近交易日
- 日期标准化：内部统一使用 `datetime.date`，自动处理 `pd.Timestamp` 和字符串

### returns — CACS 累计收益

CACS (China A-Share Cumulative-adjusting Series) 公式正确处理 A 股除权除息：

```
cum_ret = (close.e * factor.e + (const.e - const.s)) / factor.s / close.s - 1
```

其中 close.e / factor.e 为期末值，close.s / factor.s 为期初值，const.e / const.s 为调整常数。

| 函数 | 说明 |
|------|------|
| `cal_cum_ret(jtids, start_dates, end_dates, close, adj_factor)` | 计算 CACS 调整后的累计收益，返回 `np.ndarray` |

**数据规范**：

- `close`: index=date, columns=jtid  
- `adj_factor`: 可选，需含 `adjustingFactor` 和 `adjustingConst` 列

### universe — 股票池构建

| 函数 | 说明 |
|------|------|
| `build_universe(start_date, end_date, univ_type, **kwargs)` | 构建按日期的股票池 |

| 参数 | 说明 |
|------|------|
| `univ_type` | `"liquidity_based"` / `"CSI300"` / `"fixed"` |
| `univ_range` | 流动性百分位范围，默认 (0, 1) |
| `fixed_list` | 固定股票列表 (univ_type="fixed" 时必填) |
| `liquidity_data` | 日频流动性数据 (univ_type="liquidity_based" 时必填) |
| `index_constituents` | 指数成分字典 (univ_type="CSI300" 时必填) |
| `days_smooth` | 流动性平滑天数，默认 10 |

返回值：`dict[str, list[str]]`，key 为 `YYYYMMDD` 格式字符串。

## 与各子包的关系

- **ettr / eClassic / eAlpha101**：因子层可选依赖，用于按交易日对齐因子计算
- **eFactorCraft**：因子工程层可选依赖，用于数据预处理中的日期对齐
- **eBacktestCraft**：策略层可选依赖，用于回测日期范围控制和股票池筛选
- **webapp**：应用层使用，用于数据源管理中的日期范围查询

## 版本

0.1.0 — Python >= 3.9
