# eAlpha101

A Python package implementing all 101 quantitative Alpha factors from the WorldQuant paper *"101 Formulaic Alphas"* (Kakushadze, 2016). Designed for **long-format panel DataFrames** — the same API style as the companion R package `eAlpha101`.

---

## Installation

```bash
# from source
pip install -e .

# dependencies
pip install pandas>=1.5 numpy>=1.23
```

---

## Quick Start

```python
import ealpha101 as ea

# built-in sample data (10 A-share stocks × 500 trading days)
df = ea.load_sample_data()
print(df.shape)          # (5000, 28)

# compute a single alpha
result = ea.add_alpha001(df)
print(result[["date", "code", "alpha001"]].head(10))

# chain multiple alphas
df = ea.add_alpha001(df)
df = ea.add_alpha006(df)
df = ea.add_alpha101(df)
print(df[["date", "code", "alpha001", "alpha006", "alpha101"]].tail())
```

---

## Data Format

All functions accept and return a **long-format** `pd.DataFrame`.

### Required identity columns

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime-like | Trading date |
| `code` | str | Stock ticker / ID |
| `name` | str | Stock name |

### Standard price / volume columns

| Column | Default param | Description |
|--------|---------------|-------------|
| `open` | `open_col="open"` | Opening price |
| `high` | `high_col="high"` | Daily high |
| `low` | `low_col="low"` | Daily low |
| `close` | `close_col="close"` | Closing price |
| `volume` | `volume_col="volume"` | Trading volume |
| `vwap` | `vwap_col="vwap"` | Volume-weighted average price |
| `returns` | `returns_col="returns"` | Daily simple return |

### Special columns (only some alphas)

| Column | Which alphas | Description |
|--------|-------------|-------------|
| `cap` | alpha056 | Market capitalisation |
| `neut_*` | alpha048, 058, 059, 063, 067, 069, 070, 076, 079, 080, 082, 089, 091, 093, 097, 100 | Pre-industry-neutralised columns (see below) |

---

## Built-in Sample Data

```python
df = ea.load_sample_data(seed=2024)
```

Returns a 5 000-row DataFrame (10 stocks × 500 trading days) with **all 28 columns** needed to run every alpha function without extra preparation.

### Column reference

```
date, code, name,
open, high, low, close, adjusted, volume, vwap, returns,
cap, bv, op, assets, benchmark_ret,

# pre-neutralised (industry-demeaned) columns:
neut_close, neut_vwap, neut_low, neut_volume,
neut_price79, neut_price80, neut_price97, neut_vwap2,
neut_close_ret, neut_vwap_ret,
neut_rank100, neut_diff100
```

---

## Industry Neutralisation

Alphas that involve `IndNeutralize()` in the original paper **do not compute it internally**. Instead, they accept a pre-neutralised column name as a parameter. This keeps the functions pure and lets you choose your own industry classification.

```python
# Example: industry-demean vwap by date × industry
df["neut_vwap"] = (
    df["vwap"]
    - df.groupby(["date", "industry"])["vwap"].transform("mean")
)

result = ea.add_alpha058(df, neut_vwap_col="neut_vwap")
```

Alphas requiring pre-neutralised columns and their parameters:

| Alpha | Pre-neutralised param(s) |
|-------|--------------------------|
| alpha048 | `neut_close_ret_col`, `neut_vwap_ret_col` |
| alpha058 | `neut_vwap_col` |
| alpha059 | `neut_vwap2_col` |
| alpha063 | `neut_close_col` |
| alpha067 | `neut_vwap_col`, `neut_adv20_col` |
| alpha069 | `neut_vwap_col` |
| alpha070 | `neut_close_col` |
| alpha076 | `neut_low_col` |
| alpha079 | `neut_price_col` |
| alpha080 | `neut_price_col` |
| alpha082 | `neut_vol_col` |
| alpha089 | `neut_vwap_col` |
| alpha091 | `neut_close_col` |
| alpha093 | `neut_vwap_col` |
| alpha097 | `neut_price_col` |
| alpha100 | `neut_rank_col`, `neut_diff_col` |

---

## Module Structure

```
ealpha101/
├── __init__.py       # exports all 101 add_alphaXXX + load_sample_data
├── data.py           # load_sample_data()
├── _base.py          # _validate / _sort / _finish helpers
├── utils.py          # rolling utility functions
├── alpha001_020.py   # alpha001 – alpha020
├── alpha021_040.py   # alpha021 – alpha040
├── alpha041_060.py   # alpha041 – alpha060
├── alpha061_080.py   # alpha061 – alpha080
└── alpha081_101.py   # alpha081 – alpha101
```

---

## Function API

Every `add_alphaXXX` function follows the same contract:

```python
def add_alphaXXX(
    mkt_data: pd.DataFrame,
    close_col: str = "close",   # column-name params with sensible defaults
    ...
    append: bool = True,        # True → append alpha column; False → slim output
) -> pd.DataFrame:
```

**Returns** the same number of rows as the input. Row order is preserved (sorted internally by `[code, date]` then restored to original index).

```python
# append=True (default): all original columns + alpha column
result = ea.add_alpha001(df)

# append=False: only date, code, name, alphaXXX
slim = ea.add_alpha001(df, append=False)
```

---

## Utility Functions (`ealpha101.utils`)

```python
from ealpha101.utils import (
    cs_rank,        # cross-sectional percentile rank
    scale_alpha,    # scale to unit absolute sum
    ts_sum,         # rolling sum
    ts_mean,        # rolling mean
    ts_stddev,      # rolling std dev
    ts_max,         # rolling maximum
    ts_min,         # rolling minimum
    ts_rank,        # rolling rank (percentile)
    ts_argmax,      # rolling argmax position (1-indexed)
    ts_argmin,      # rolling argmin position (1-indexed)
    ts_product,     # rolling product
    delay,          # lag / shift
    delta,          # difference from lag
    correlation,    # rolling Pearson correlation
    covariance,     # rolling covariance
    decay_linear,   # linearly weighted moving average
    signedpower,    # sign(x) * abs(x)^exp
    adv,            # d-day average daily volume
)
```

---

## Examples

### Example 1: Basic usage with built-in data

```python
import ealpha101 as ea

df = ea.load_sample_data()

# alpha101: (close - open) / (high - low + 0.001) — intraday momentum
result = ea.add_alpha101(df)
print(result[["date", "code", "alpha101"]].head())
```

### Example 2: Multiple alphas in a pipeline

```python
df = ea.load_sample_data()

for add_fn in [ea.add_alpha001, ea.add_alpha006, ea.add_alpha011,
               ea.add_alpha025, ea.add_alpha101]:
    df = add_fn(df)

alpha_cols = [c for c in df.columns if c.startswith("alpha")]
print(df[["date", "code"] + alpha_cols].tail())
```

### Example 3: Slim output for factor storage

```python
df = ea.load_sample_data()
factor = ea.add_alpha001(df, append=False)
# factor has only: date, code, name, alpha001
print(factor.shape)   # (5000, 4)
```

### Example 4: Alphas requiring pre-neutralised columns

```python
df = ea.load_sample_data()
# neut_vwap is already pre-computed in the sample data
result = ea.add_alpha058(df, neut_vwap_col="neut_vwap", volume_col="volume")
print(result[["date", "code", "alpha058"]].dropna().head())
```

### Example 5: Custom column names

```python
# your DataFrame uses different column names
result = ea.add_alpha001(
    my_df,
    close_col="price_close",
    returns_col="daily_return",
)
```

### Example 6: Cross-sectional IC analysis

```python
df = ea.load_sample_data()
df = ea.add_alpha001(df, append=True)

# forward 1-day return
df["fwd_ret"] = df.groupby("code")["returns"].shift(-1)

# daily rank IC (Spearman)
ic = (
    df.dropna(subset=["alpha001", "fwd_ret"])
    .groupby("date")
    .apply(lambda g: g["alpha001"].corr(g["fwd_ret"], method="spearman"))
)
print(f"Mean IC: {ic.mean():.4f}  |  ICIR: {ic.mean()/ic.std():.4f}")
```

### Example 7: alpha056 (requires cap column)

```python
df = ea.load_sample_data()
# cap column is included in sample data
result = ea.add_alpha056(df, returns_col="returns", cap_col="cap")
print(result[["date", "code", "alpha056"]].head())
```

---

## Alpha Reference

| Module | Alphas | Key inputs |
|--------|--------|-----------|
| alpha001_020.py | 001–020 | close, open, high, low, volume, vwap, returns |
| alpha021_040.py | 021–040 | close, open, high, low, volume, vwap, returns |
| alpha041_060.py | 041–060 | close, open, high, low, volume, vwap, returns, cap (056 only) |
| alpha061_080.py | 061–080 | close, open, high, low, volume, vwap + neut_* columns |
| alpha081_101.py | 081–101 | close, open, high, low, volume, vwap + neut_* columns |

---

## Reference

Kakushadze, Z. (2016). *101 Formulaic Alphas*. Wilmott Magazine, 2016(84), 72–81.  
[https://arxiv.org/abs/1601.00991](https://arxiv.org/abs/1601.00991)

---

## See Also

- **eAlpha101** (R package) — same 101 alphas with identical long-format API
- **eClassic** (R package) — Fama-French 6-factor model functions (Size, Value, Profitability, Investment, Beta, Momentum)

---

## 联系我们

| | |
|---|---|
| 🌐 公司官网 | [xquant.shop](https://xquant.shop) |
| 📱 公司公众号 | xquant-shop |
| 📱 个人公众号 | i锐角 |