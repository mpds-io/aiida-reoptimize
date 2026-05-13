"""Shared helpers for static optimization workchains."""

from collections.abc import Callable
from typing import Any

from aiida.orm import Str

from ...base.Extractors import BasicExtractor
from ...optimizers.convex.GD import (
    AdamOptimizer,
    ConjugateGradientOptimizer,
    RMSpropOptimizer,
)
from ...optimizers.convex.QN import BFGSOptimizer
from ...optimizers.PyMOO.PyMOO import PyMOO_Optimizer


def output_path_extractor(path: tuple[str, ...]) -> Callable[[Any], Any]:
    """Build a node-output extractor that walks a nested key path.

    Args:
        path: Tuple of attribute/key names to traverse (e.g. ``("output_scf_wc_para", "total_energy")``).

    Returns:
        A callable that, given an AiiDA node's ``outputs``, returns the value at the path.
    """

    def _extract(outputs: Any) -> Any:
        value = outputs
        for key in path:
            value = value[key]
        return value

    return _extract


class StaticOptimizerBinding:
    """Mixin that binds evaluator workchain and extractor for static optimizers.

    Sets ``evaluator_workchain`` and creates a ``BasicExtractor`` from the
    ``extractor_path`` tuple when a subclass is created.
    """

    evaluator_workchain = None
    extractor_path: tuple[str, ...] = ()
    extractor = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.extractor_path:
            cls.extractor = BasicExtractor(node_extractor=output_path_extractor(cls.extractor_path))


class FixedPyMOOAlgorithmMixin:
    """Mixin that pins a static optimizer WorkChain to a single PyMOO algorithm.

    Subclasses must set the ``fixed_algorithm_name`` class attribute to the
    desired algorithm string (e.g. ``"G3PCX"`` or ``"NRBO"``).
    """

    fixed_algorithm_name = ""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.inputs["algorithm_name"].default = lambda: Str(cls.fixed_algorithm_name)
        spec.inputs["algorithm_name"].help = f"Fixed PyMOO algorithm name ({cls.fixed_algorithm_name})."

    def initialize(self):
        exit_code = super().initialize()
        if exit_code is not None:
            return exit_code
        self.ctx.algorithm_name = self.fixed_algorithm_name


__all__ = [
    "AdamOptimizer",
    "BFGSOptimizer",
    "ConjugateGradientOptimizer",
    "FixedPyMOOAlgorithmMixin",
    "PyMOO_Optimizer",
    "RMSpropOptimizer",
    "StaticOptimizerBinding",
]
