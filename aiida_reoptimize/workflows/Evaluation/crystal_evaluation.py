from aiida_crystal_dft.workflows.base import BaseCrystalWorkChain

from ...base.Evaluation import StaticEvalLatticeProblem


class CrystalLatticeProblem(StaticEvalLatticeProblem):
    calculator_workchain = BaseCrystalWorkChain