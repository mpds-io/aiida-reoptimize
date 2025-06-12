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


# Defining the exctractor
def extractor(results: list, node_extractor: callable, penalty=1e10):
    """Basic function that extracts data after evaluations.
    results: list
        Expected to be a list of dictionaries, each containing:
        - status: Either "ok" or "failed"
        - pk: integer value
    node_extractor: callable
        A callable that takes a node as input and returns a value to be 
        appended to `values`.
        (This design allows for flexibility in handling nested node outputs)
    penalty: float
        Value that will be used as a fallback when a node cannot be loaded
        or when the status is not "ok", indicating a failed calculation.
    """
    values = []
    for item in results:
        if item["status"] == "ok":
            try:
                node = load_node(item["pk"])
                value = node_extractor(node.outputs)
            except NotExistent:
                value = penalty
        else:
            value = penalty
        values.append(value)
    return values


results = run(UserEvaluator, targets=targets)
print(results["evaluation_results"])

data = extractor(
    results=results["evaluation_results"],
    node_extractor=lambda x: x["output_scf_wc_para"]["total_energy"],
)

print(data)
