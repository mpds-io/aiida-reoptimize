"""Static evaluator workchains for CRYSTAL-based lattice calculations."""

from aiida_crystal_dft.workflows.base import BaseCrystalWorkChain

from ...base.Evaluation import StaticEvalLatticeProblem


class CrystalLatticeProblem(StaticEvalLatticeProblem):
    """Evaluate lattice perturbations with the CRYSTAL base workchain.

    Inputs inherited from :class:`StaticEvalLatticeProblem`:
    - ``structure``: source crystal structure.
    - ``targets``: list of lattice parameter perturbations.
    - ``calculator_parameters``: inputs forwarded to CRYSTAL.
    - ``structure_keyword``: path to the structure input in the builder.
    """

    calculator_workchain = BaseCrystalWorkChain
