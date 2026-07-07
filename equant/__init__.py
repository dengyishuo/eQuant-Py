"""
eQuant: Unified quantitative research toolkit.

Sub-packages
------------
ettr          Technical trading rules and indicators
eclassic      Classic quantitative indicators (momentum, RPS, RAM, ...)
efactorcraft  Factor data acquisition and preprocessing
ebacktestcraft  Backtesting framework
ealpha101     WorldQuant 101 alpha factors
"""

from importlib.metadata import version, PackageNotFoundError

def _ver(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not installed"

__version__ = _ver("eQuant")

# Lazy imports — only load sub-packages when accessed
def __getattr__(name: str):
    _map = {
        "utils":          "equant.utils",
        "ettr":           "ettr",
        "eclassic":       "eclassic",
        "efactorcraft":   "efactorcraft",
        "ebacktestcraft": "ebacktestcraft",
        "ealpha101":      "ealpha101",
    }
    if name in _map:
        import importlib
        mod = importlib.import_module(_map[name])
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'equant' has no attribute {name!r}")


def versions() -> dict:
    """Print version of each sub-package."""
    pkgs = ["eTTR", "eClassic", "eFactorCraft", "eBacktestCraft", "eAlpha101", "eQuant"]
    return {p: _ver(p) for p in pkgs}
