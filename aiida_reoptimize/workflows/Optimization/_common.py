"""Shared helpers for static optimization workchains."""

from collections.abc import Callable
from typing import Any

from ...base.Extractors import BasicExtractor
from ...optimizers.convex.GD import (
    AdamOptimizer,
    ConjugateGradientOptimizer,
    RMSpropOptimizer,
)
from ...optimizers.convex.QN import BFGSOptimizer
from ...optimizers.PyMOO.PyMOO import PyMOO_Optimizer


def output_path_extractor(path: tuple[str, ...]) -> Callable[[Any], Any]:
    """Build a node-output extractor that walks a nested key path."""

    def _extract(outputs: Any) -> Any:
        value = outputs
        for key in path:
            value = value[key]
        return value

    return _extract


class StaticOptimizerBinding:
    """Mixin that binds evaluator workchain and extractor for static optimizers."""

    evaluator_workchain = None
    extractor_path: tuple[str, ...] = ()
    extractor = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.extractor_path:
            cls.extractor = BasicExtractor(node_extractor=output_path_extractor(cls.extractor_path))


__all__ = [
    "AdamOptimizer",
    "BFGSOptimizer",
    "ConjugateGradientOptimizer",
    "PyMOO_Optimizer",
    "RMSpropOptimizer",
    "StaticOptimizerBinding",
]
