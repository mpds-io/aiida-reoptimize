import aiida
from aiida.engine import run
from aiida.orm import (
    Dict,
    Int,
    Str,
    load_node,
)
from aiida_fleur.workflows.scf import FleurScfWorkChain
from ase import Atoms

from aiida_reoptimize.base.Evaluation import EvalWorkChainStructureProblem
from aiida_reoptimize.optimizers.PyMOO.PyMOO import PyMOO_Optimizer
from aiida_reoptimize.structure.dynamic_structure import StructureCalculator

aiida.load_profile()

a = 5.51
c = 7.79
# Create unit cell matrix
cell = [
    [-0.5 * a, 0.5 * a, 0.5 * c],
    [0.5 * a, -0.5 * a, 0.5 * c],
    [0.5 * a, 0.5 * a, -0.5 * c],
]

# Atomic positions (fractional coordinates)
scaled_positions = [
    (0.75, 0.25, 0.5),  # Sr
    (0.25, 0.75, 0.5),  # Sr
    (0.0, 0.0, 0.0),  # Ti
    (0.5, 0.5, 0.0),  # Ti
    (0.25894899, 0.24105101, 0.5),  # O
    (0.74105101, 0.75894899, 0.5),  # O
    (0.24105101, 0.74105101, 0.98210202),  # O
    (0.75894899, 0.25894899, 0.01789798),  # O
    (0.25, 0.25, 0.0),  # O
    (0.75, 0.75, 0.0),  # O
]

# Create ASE Atoms object for structure
symbols = ["Sr", "Sr", "Ti", "Ti", "O", "O", "O", "O", "O", "O"]
initial_structure = Atoms(
    symbols=symbols,
    scaled_positions=scaled_positions,
    cell=cell,
    pbc=True,
)

# Define the generator for the Evaluator
problem_builder = StructureCalculator(
    structure=initial_structure,
    calculator=FleurScfWorkChain,
    parameters={"inpgen": load_node(101745), "fleur": load_node(101746)},
    structure_keyword="structure",
)


# Define Evaluator
class UserEvaluator(EvalWorkChainStructureProblem):
    problem_builder = problem_builder
    extractor = staticmethod(lambda x: x["output_scf_wc_para"]["total_energy"])  # noqa: E731


class ExamplePyMOO(PyMOO_Optimizer):
    evaluator_workchain = UserEvaluator


parameters = Dict({
    "dimensions": 2,
    "bounds": [[a - a * 0.1, a + a * 0.1], [c - c * 0.1, c + c * 0.1]],
    "algorithm_settings": {"pop_size": 2},
})

__parameters = Dict(
    dict={
        "itmax": Int(2),
        "parameters": parameters,
        "algorithm_name": Str("PSO"),
    }
)

results = run(
    ExamplePyMOO,
    **__parameters,
)

print("Optimization Results:")
print(f"Best position: {results['optimized_parameters']}")
print(f"Best value: {results['final_value']}")
