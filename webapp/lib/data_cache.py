"""Local parquet cache for efactorcraft.get_data() results.

Keyed by (source, sorted codes, start_date, end_date) so re-visiting the
same universe/date range/source skips the network round-trip.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

CACHE_DIR = Path.home() / ".equant" / "cache"


def _cache_key(source: str, universe: pd.DataFrame, start_date: str, end_date: str) -> str:
    codes = ",".join(sorted(universe["code"].astype(str)))
    raw = f"{source}|{codes}|{start_date}|{end_date}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached(source: str, universe: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{_cache_key(source, universe, start_date, end_date)}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def set_cached(source: str, universe: pd.DataFrame, start_date: str, end_date: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(source, universe, start_date, end_date)}.parquet"
    df.to_parquet(path)
