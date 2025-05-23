import aiida
from aiida.engine import run
from aiida.orm import (
    List,
    load_node,
)
from aiida_fleur.workflows.scf import FleurScfWorkChain
from ase.build import bulk

from src.base.Evaluation import EvalWorkChainStructureProblem
from src.structure.dynamic_structure import StructureCalculator

aiida.load_profile()


# Example usage
initial_structure = bulk("Au")

# Define the generator for the Evaluator
problem_builder = StructureCalculator(
    structure=initial_structure,
    calculator=FleurScfWorkChain,
    parameters={"inpgen": load_node(101745), "fleur": load_node(101746)},
    structure_keyword="structure",
)

targets = List(list=[[4.2], [4.3]])  # Example list of x values


# Define Evaluator
class UserEvaluator(EvalWorkChainStructureProblem):
    problem_builder = problem_builder
    extractor = staticmethod(
        lambda x: x["output_scf_wc_para"]["total_energy"]
    )  # noqa: E731


results = run(UserEvaluator, targets=targets)
print(results["evaluation_results"])
