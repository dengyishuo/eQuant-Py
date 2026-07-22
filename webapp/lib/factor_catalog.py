"""Builds a UI-facing factor catalog by introspecting eclassic / ettr / ealpha101.

Every factor function in these packages follows the same convention: first
positional arg is the panel DataFrame, remaining args are simple scalars
(column names, window lengths, flags) with defaults, last is ``append``.
That regularity means one generic `inspect.signature` walk can drive the
Streamlit form for all ~90 factors instead of hand-writing a form per factor.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

import lib.bootstrap  # noqa: F401  (must run before importing e* packages)

import eclassic
import ettr
import ealpha101
from ealpha101.catalog import ALPHAS

BASE_COLS = {"date", "code", "name", "open", "high", "low", "close", "adjusted", "volume"}

# Utility functions that don't follow the "df in, df out, scalar params" factor
# convention (extra required DataFrame arg, or return a dict instead of a
# DataFrame) — not selectable as factors in the UI.
_ETTR_EXCLUDE = {"align_with_index", "calculate_performance"}


@dataclass
class ParamSpec:
    name: str
    kind: str  # "int" | "float" | "bool" | "int_list" | "str"
    default: Any
    required: bool


@dataclass
class FactorSpec:
    package: str
    name: str
    fn: Callable
    params: list[ParamSpec] = field(default_factory=list)
    required_cols: list[str] | None = None  # None = unknown, assume OHLCV suffices


def _param_specs(fn: Callable) -> list[ParamSpec]:
    sig = inspect.signature(fn)
    specs = []
    for i, (pname, p) in enumerate(sig.parameters.items()):
        if i == 0 or pname == "append":
            continue
        has_default = p.default is not inspect.Parameter.empty
        default = p.default if has_default else None
        if isinstance(default, bool):
            kind = "bool"
        elif isinstance(default, (tuple, list)):
            kind = "int_list"
        elif isinstance(default, int):
            kind = "int"
        elif isinstance(default, float):
            kind = "float"
        else:
            kind = "str"
        specs.append(ParamSpec(name=pname, kind=kind, default=default, required=not has_default))
    return specs


def _module_catalog(package: str, module) -> list[FactorSpec]:
    entries = []
    for name in module.__all__:
        if package == "ettr" and name in _ETTR_EXCLUDE:
            continue
        fn = getattr(module, name)
        if not callable(fn):
            continue
        entries.append(FactorSpec(package=package, name=name, fn=fn, params=_param_specs(fn)))
    return entries


def _alpha_catalog() -> list[FactorSpec]:
    entries = []
    for name, meta in ALPHAS.items():
        fn = getattr(ealpha101, name, None)
        if fn is None:
            continue
        params = [
            ParamSpec(name=k, kind="str", default=v, required=False)
            for k, v in meta["params"].items()
        ]
        entries.append(
            FactorSpec(package="ealpha101", name=name, fn=fn, params=params, required_cols=meta["required"])
        )
    return entries


def build_catalog() -> dict[str, list[FactorSpec]]:
    return {
        "eclassic": _module_catalog("eclassic", eclassic),
        "ettr": _module_catalog("ettr", ettr),
        "ealpha101": _alpha_catalog(),
    }


def available_factors(columns: set[str]) -> dict[str, list[FactorSpec]]:
    """Filter ealpha101 entries to those whose required columns are already
    present; eclassic/ettr entries are always offered since their column
    params auto-detect from close/adjusted (present in every panel)."""
    catalog = build_catalog()
    cols = set(columns) | BASE_COLS
    catalog["ealpha101"] = [e for e in catalog["ealpha101"] if set(e.required_cols or []) <= cols]
    return catalog
