"""Static optimization workchains for FLEUR SCF lattice problems."""

from ..Evaluation.fleur_evaluators import FleurSCFLatticeProblem
from ._common import (
    AdamOptimizer,
    BFGSOptimizer,
    ConjugateGradientOptimizer,
    FixedPyMOOAlgorithmMixin,
    PyMOO_Optimizer,
    RMSpropOptimizer,
    StaticOptimizerBinding,
)


class BaseFleurSCFOptimizer(StaticOptimizerBinding):
    """Bind the generic optimizers to the FLEUR SCF lattice evaluator."""

    evaluator_workchain = FleurSCFLatticeProblem
    extractor_path = ("output_scf_wc_para", "total_energy")


class AdamFleurSCFOptimizer(BaseFleurSCFOptimizer, AdamOptimizer):
    """Adam optimizer registered for FLEUR SCF lattice optimization."""


class CDGFleurSCFOptimizer(BaseFleurSCFOptimizer, ConjugateGradientOptimizer):
    """Conjugate-gradient optimizer registered for FLEUR SCF lattice optimization."""


class RMSpropFleurSCFOptimizer(BaseFleurSCFOptimizer, RMSpropOptimizer):
    """RMSprop optimizer registered for FLEUR SCF lattice optimization."""


class BFGSFleurSCFOptimizer(BaseFleurSCFOptimizer, BFGSOptimizer):
    """BFGS optimizer registered for FLEUR SCF lattice optimization."""


class PyMOOFleurSCFOptimizer(BaseFleurSCFOptimizer, PyMOO_Optimizer):
    """PyMOO-backed optimizer registered for FLEUR SCF lattice optimization."""


class G3PCXFleurSCFOptimizer(
    FixedPyMOOAlgorithmMixin,
    BaseFleurSCFOptimizer,
    PyMOO_Optimizer,
):
    """Fixed G3PCX PyMOO optimizer for FLEUR SCF lattice optimization."""

    fixed_algorithm_name = "G3PCX"


class NRBOFleurSCFOptimizer(
    FixedPyMOOAlgorithmMixin,
    BaseFleurSCFOptimizer,
    PyMOO_Optimizer,
):
    """Fixed NRBO PyMOO optimizer for FLEUR SCF lattice optimization."""

    fixed_algorithm_name = "NRBO"
