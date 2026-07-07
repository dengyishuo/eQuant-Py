"""Type aliases for quantkit's long-format panel data convention."""

from __future__ import annotations

import sys

import pandas as pd

# Core type: long-format panel DataFrame
# Every quantkit function expects: date, code, name columns as identifiers,
# plus arbitrary OHLCV/factor columns. One row = one asset-date observation.
if sys.version_info >= (3, 10):
    from typing import TypeAlias
    PanelFrame: TypeAlias = pd.DataFrame
else:
    PanelFrame = pd.DataFrame  # type: ignore
