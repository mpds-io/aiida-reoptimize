from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, Int, Str

from aiida_reoptimize.base.Evaluation import EvalWorkChainProblem
from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.optimizers.PyMOO.PyMOO import PyMOO_Optimizer
from aiida_reoptimize.problems.problems import Sphere

load_profile()

# Setup extractor
dummy_extractor = BasicExtractor(node_extractor=lambda x: x["value"])


# Setup evaluator
class UserEvaluator(EvalWorkChainProblem):
    problem_workchain = Sphere


class ExamplePyMOO(PyMOO_Optimizer):
    evaluator_workchain = UserEvaluator
    extractor = dummy_extractor


parameters = Dict(
    {
        "bounds": [[-1.0, 3.0], [-5.0, 4.0], [-2.0, 1.0]],
        "tol": 1e-4,
        "algorithm_settings": {
            "pop_size": 20,
            "sampling": "LHS",
            "max_iteration": 200,
            "deciding_factor": 0.6,
        },
    }
)

optimizer_inputs = {
    "itmax": Int(30),
    "parameters": parameters,
    "algorithm_name": Str("NRBO"),
}

results = run(
    ExamplePyMOO,
    **optimizer_inputs,
)

print("Optimization Results:")
if results:
    print(results)
    print(f"Best position: {results['optimized_parameters']}")
    print(f"Best value: {results['final_value']}")
    print(f"Best node: {results['result_node_pk']}")

print("Optimization history:")
for iter_ in results["history"]:
    print(iter_)
