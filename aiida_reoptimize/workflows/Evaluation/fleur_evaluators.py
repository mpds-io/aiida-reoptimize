"""Static evaluator workchains for FLEUR-based lattice calculations."""

from aiida_fleur.workflows.relax import FleurRelaxWorkChain
from aiida_fleur.workflows.scf import FleurScfWorkChain

from ...base.Evaluation import StaticEvalLatticeProblem


class FleurSCFLatticeProblem(StaticEvalLatticeProblem):
    """Evaluate lattice perturbations with the FLEUR SCF workchain.

    The workflow accepts the generic static evaluator inputs and forwards the
    resolved builder parameters to ``FleurScfWorkChain``.
    """

    calculator_workchain = FleurScfWorkChain


class FleurRelaxLatticeProblem(StaticEvalLatticeProblem):
    """Evaluate lattice perturbations with the FLEUR relax workchain.

    The workflow accepts the generic static evaluator inputs and forwards the
    resolved builder parameters to ``FleurRelaxWorkChain``.
    """

    calculator_workchain = FleurRelaxWorkChain
