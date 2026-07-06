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
# From My-Pkg root — install all sub-packages in editable mode
pip install -e eTTR-Py -e eClassic-Py -e eFactorCraft-Py -e eBacktestCraft-Py -e eAlpha101-Py -e eQuant-Py
```

## Usage

```python
import equant
print(equant.versions())

# Sub-packages are lazy-loaded on first access
from ealpha101 import Alpha
from eclassic import add_rps
```
