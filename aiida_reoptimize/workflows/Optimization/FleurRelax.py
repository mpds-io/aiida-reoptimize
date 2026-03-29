"""Static optimization workchains for FLEUR relax lattice problems."""

from ..Evaluation.fleur_evaluators import FleurRelaxLatticeProblem
from ._common import (
    AdamOptimizer,
    BFGSOptimizer,
    ConjugateGradientOptimizer,
    PyMOO_Optimizer,
    RMSpropOptimizer,
    StaticOptimizerBinding,
)


class BaseFleurRelaxOptimizer(StaticOptimizerBinding):
    """Bind the generic optimizers to the FLEUR relax lattice evaluator."""

    evaluator_workchain = FleurRelaxLatticeProblem
    extractor_path = ("output_relax_wc_para", "energy")


class AdamFleurRelaxOptimizer(BaseFleurRelaxOptimizer, AdamOptimizer):
    """Adam optimizer registered for FLEUR relax lattice optimization."""


class CDGFleurRelaxOptimizer(BaseFleurRelaxOptimizer, ConjugateGradientOptimizer):
    """Conjugate-gradient optimizer registered for FLEUR relax lattice optimization."""


class RMSpropFleurRelaxOptimizer(BaseFleurRelaxOptimizer, RMSpropOptimizer):
    """RMSprop optimizer registered for FLEUR relax lattice optimization."""


class BFGSFleurRelaxOptimizer(BaseFleurRelaxOptimizer, BFGSOptimizer):
    """BFGS optimizer registered for FLEUR relax lattice optimization."""


class PyMOOFleurRelaxOptimizer(BaseFleurRelaxOptimizer, PyMOO_Optimizer):
    """PyMOO-backed optimizer registered for FLEUR relax lattice optimization."""
