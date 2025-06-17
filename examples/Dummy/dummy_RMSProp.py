from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, Int, List

from aiida_reoptimize.base.Evaluation import EvalWorkChainProblem
from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.optimizers.convex.GD import RMSpropOptimizer
from aiida_reoptimize.problems.problems import Sphere

load_profile()


# Setup basic extractor
dummy_extractor = BasicExtractor(node_extractor=lambda x: x["value"])


# setup Evaluator
class UserEvaluator(EvalWorkChainProblem):
    problem_workchain = Sphere


class ExampleRMSprop(RMSpropOptimizer):
    evaluator_workchain = UserEvaluator
    extractor = dummy_extractor


parameters = Dict({
    "algorithm_settings": {"learning_rate": 0.1, "rho": 0.5},
    "initial_parameters": List([1.1, -5.1, -3.2]),
})

__parameters = Dict(dict={
    "itmax": Int(100),
    "parameters": parameters,
})

results = run(
    ExampleRMSprop,
    **__parameters,
)

print("Optimization Results:")
if results:
    print(results)
    print(f"Best position: {results['optimized_parameters']}")
    print(f"Best value: {results['final_value']}")
    print(f"Best node: {results['result_node_pk']}")

print("Optimization history:")
for iter_ in results['history']:
    print(iter_)
