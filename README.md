# eQuant

Unified quantitative research toolkit. Installing `eQuant` pulls in all sub-packages at once.

## Sub-packages

| Package | Import | Description |
|---------|--------|-------------|
| eTTR | `import ettr` | Technical trading rules and indicators |
| eClassic | `import eclassic` | Classic indicators: momentum, RPS, RAM, etc. |
| eFactorCraft | `import efactorcraft` | Factor data acquisition and preprocessing |
| eBacktestCraft | `import ebacktestcraft` | Backtesting framework |
| eAlpha101 | `import ealpha101` | WorldQuant 101 alpha factors |

## Install

```bash
# Install all at once (recommended)
pip install eQuant

# Or install individually
pip install eTTR
pip install eClassic
```

## Development install

```bash
# From repo root — install all sub-packages in editable mode
pip install -e eTTR-Py -e eClassic-Py -e eFactorCraft-Py -e eBacktestCraft-Py -e eAlpha101-Py
pip install -e .  # umbrella equant package
```

## Usage

```python
import equant
print(equant.versions())

# Sub-packages are lazy-loaded on first access
from ealpha101 import ALPHAS
from ettr import sma, ema, rsi
```

## Tests

```bash
# Run all tests (requires PYTHONPATH to find sub-packages)
PYTHONPATH=eTTR-Py:eClassic-Py:eAlpha101-Py:eFactorCraft-Py:eBacktestCraft-Py:. \
  python3 -m pytest eTTR-Py/tests eClassic-Py/tests eAlpha101-Py/tests eFactorCraft-Py/tests eBacktestCraft-Py/tests -v
```
