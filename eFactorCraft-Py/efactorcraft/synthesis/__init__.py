"""Factor synthesis — combine multiple factor columns into composite scores."""

from efactorcraft.synthesis.composite import (
    equal_weighted_composite,
    ic_weighted_composite,
    icir_weighted_composite,
    max_decay_composite,
    pca_composite,
    rank_weighted_composite,
)

__all__ = [
    "equal_weighted_composite",
    "ic_weighted_composite",
    "icir_weighted_composite",
    "max_decay_composite",
    "pca_composite",
    "rank_weighted_composite",
]
