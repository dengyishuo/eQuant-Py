"""eClassic — Classic Fama-French style factors for long-format panel DataFrames.

Each function takes a long-format panel DataFrame and returns it with
new factor columns appended.

Usage::

    from eclassic import momentum, value, size

    df = momentum(df, close_col="adjusted", n=[5, 10, 20])
    df = value(df, bv_col="bv", cap_col="cap")
    df = size(df, cap_col="cap")
    df = beta(df, close_col="close", benchmark_col="bench_close", n=60)
"""

from eclassic.benchmark import benchmark
from eclassic.beta import beta, slope
from eclassic.investment import investment
from eclassic.momentum import momentum
from eclassic.profitability import profitability
from eclassic.ram import ram
from eclassic.return_ import return_
from eclassic.rps import rps
from eclassic.size import size
from eclassic.sma import sma
from eclassic.value import value
from eclassic.volatility import volatility

__all__ = [
    "momentum",
    "value",
    "size",
    "beta",
    "slope",
    "volatility",
    "profitability",
    "investment",
    "return_",
    "sma",
    "ram",
    "rps",
    "benchmark",
]
__version__ = "0.1.0"
