from aiida_fleur.workflows.relax import FleurRelaxWorkChain
from aiida_fleur.workflows.scf import FleurScfWorkChain

from ...base.Evaluation import StaticEvalLatticeProblem


class FleurSCFLatticeProblem(StaticEvalLatticeProblem):
    calculator_workchain = FleurScfWorkChain


class FleurRelaxLatticeProblem(StaticEvalLatticeProblem):
    calculator_workchain = FleurRelaxWorkChain