from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, Int, List

from aiida_reoptimize.base.Evaluation import EvalWorkChainProblem
from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.optimizers.convex.QN import BFGSOptimizer
from aiida_reoptimize.problems.problems import Sphere

load_profile()

# Setup extracor
dummy_extractor = BasicExtractor(node_extractor=lambda x: x["value"])


# setup Evaluator
class UserEvaluator(EvalWorkChainProblem):
    problem_workchain = Sphere


class ExampleBFGS(BFGSOptimizer):
    evaluator_workchain = UserEvaluator
    extractor = dummy_extractor


parameters = Dict({
    "algorithm_settings": {
        "alpha": 1,
        "beta": 0.8,
        "sigma": 1e-4,
        "linesearch_max_iter": 20,
        "delta": 5e-4},
    "initial_parameters": List([0.3, -0.2, -0.4]),
})

__parameters = Dict(
    dict={
        "itmax": Int(100),
        "parameters": parameters,
    }
)

results = run(
    ExampleBFGS,
    **__parameters,
)

print("Optimization Results:")
if results:
    print(f"Best position: {results['optimized_parameters']}")
    print(f"Best value: {results['final_value']}")
    print(f"Best node: {results['result_node_pk']}")

print("Optimization history:")
for iter_ in results['history']:
    print(iter_)
