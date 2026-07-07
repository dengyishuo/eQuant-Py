"""Strategy enhancement — improved weights, signals, and risk controls."""

from ebacktestcraft.enhance.weights import (
    confidence_weight,
    erp_weight,
    target_vol_weight,
    vol_parity_weight,
)
from ebacktestcraft.enhance.signals import (
    persistent_signal,
    quantile_signal,
    smoothed_signal,
)
from ebacktestcraft.enhance.risk import (
    apply_vol_target,
    compute_turnover,
)

__all__ = [
    "vol_parity_weight",
    "target_vol_weight",
    "erp_weight",
    "confidence_weight",
    "quantile_signal",
    "persistent_signal",
    "smoothed_signal",
    "apply_vol_target",
    "compute_turnover",
]
