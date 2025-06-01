import aiida
from aiida.common.exceptions import NotExistent
from aiida.engine import run
from aiida.orm import (
    List,
    load_node,
)
from aiida_fleur.workflows.scf import FleurScfWorkChain
from ase.build import bulk

from aiida_reoptimize.base.Evaluation import EvalWorkChainStructureProblem
from aiida_reoptimize.base.utils import find_nodes
from aiida_reoptimize.structure.dynamic_structure import StructureCalculator

aiida.load_profile()

fleur_node_label, inpgen_node_label = "fleur", "inpgen"
nodes = find_nodes(fleur_node_label, inpgen_node_label)
required_codes = [fleur_node_label, inpgen_node_label]

for code_label in required_codes:
    if code_label not in nodes:
        raise KeyError(f"Missing required code: {code_label}")
    
    try:
        fleur_code = load_node(nodes[fleur_node_label])
        inpgen_code = load_node(nodes[inpgen_node_label])
    except NotExistent as e:
        raise RuntimeError(f"Failed to load code node: {e}") from e

# Example usage
initial_structure = bulk("Au")

# Define the generator for the Evaluator
problem_builder = StructureCalculator(
    structure=initial_structure,
    calculator=FleurScfWorkChain,
    parameters={"inpgen": inpgen_code, "fleur": fleur_code},
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
