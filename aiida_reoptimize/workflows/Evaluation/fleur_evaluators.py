"""Static evaluator workchains for FLEUR-based lattice calculations."""

from aiida.orm import List
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
    resolved builder parameters to ``FleurRelaxWorkChain``. aiida-fleur 2.x
    exposes the calculator inputs (``fleur``, ``inpgen``, ``structure``) under
    the ``scf`` namespace, so resolved code inputs are moved there and the
    generated structure is injected at ``scf.structure``.
    """

    calculator_workchain = FleurRelaxWorkChain

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "structure_keyword",
            valid_type=List,
            required=False,
            default=lambda: List(["scf", "structure"]),
            help="Path to the structure input in the calculator builder.",
        )

    def prepare_calculator_parameters(self):
        """Move resolved code inputs into the ``scf`` namespace of the calculator."""
        calculator_parameters = super().prepare_calculator_parameters()
        scf = calculator_parameters.setdefault("scf", {})
        for key in ("fleur", "inpgen"):
            if key in calculator_parameters and key not in scf:
                scf[key] = calculator_parameters.pop(key)
        return calculator_parameters
