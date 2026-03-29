"""Static optimization workchains for CRYSTAL-based lattice problems."""

from ..Evaluation.crystal_evaluation import CrystalLatticeProblem
from ._common import (
    AdamOptimizer,
    BFGSOptimizer,
    ConjugateGradientOptimizer,
    PyMOO_Optimizer,
    RMSpropOptimizer,
    StaticOptimizerBinding,
)


class BaseCrystalOptimizer(StaticOptimizerBinding):
    """Bind the generic optimizers to the CRYSTAL lattice evaluator."""

    evaluator_workchain = CrystalLatticeProblem
    extractor_path = ("output_parameters", "energy")


class AdamCrystalOptimizer(BaseCrystalOptimizer, AdamOptimizer):
    """Adam optimizer registered for CRYSTAL lattice optimization."""


class CDGCrystalOptimizer(BaseCrystalOptimizer, ConjugateGradientOptimizer):
    """Conjugate-gradient optimizer registered for CRYSTAL lattice optimization."""


class RMSpropCrystalOptimizer(BaseCrystalOptimizer, RMSpropOptimizer):
    """RMSprop optimizer registered for CRYSTAL lattice optimization."""


class BFGSCrystalOptimizer(BaseCrystalOptimizer, BFGSOptimizer):
    """BFGS optimizer registered for CRYSTAL lattice optimization."""


class PyMOOCrystalOptimizer(BaseCrystalOptimizer, PyMOO_Optimizer):
    """PyMOO-backed optimizer registered for CRYSTAL lattice optimization."""
