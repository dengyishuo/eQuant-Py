"""Local persistence under ~/.equant/: data-source config and saved strategies.

Tushare tokens and other local credentials live only in config.json on disk
— never committed to git, never sent anywhere but the configured data source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EQUANT_HOME = Path.home() / ".equant"
CONFIG_PATH = EQUANT_HOME / "config.json"
STRATEGIES_DIR = EQUANT_HOME / "strategies"


def _ensure_dirs() -> None:
    EQUANT_HOME.mkdir(exist_ok=True)
    STRATEGIES_DIR.mkdir(exist_ok=True)


def load_config() -> dict[str, Any]:
    _ensure_dirs()
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def save_config(cfg: dict[str, Any]) -> None:
    _ensure_dirs()
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def list_strategies() -> list[str]:
    _ensure_dirs()
    return sorted(p.stem for p in STRATEGIES_DIR.glob("*.json"))


def load_strategy(name: str) -> dict[str, Any]:
    path = STRATEGIES_DIR / f"{name}.json"
    return json.loads(path.read_text())


def save_strategy(name: str, data: dict[str, Any]) -> None:
    _ensure_dirs()
    path = STRATEGIES_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def delete_strategy(name: str) -> None:
    path = STRATEGIES_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
