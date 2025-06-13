from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, Int, List

from aiida_reoptimize.base.Evaluation import EvalWorkChainProblem
from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.optimizers.convex.GD import AdamOptimizer
from aiida_reoptimize.problems.problems import Sphere

load_profile()

# Setup basic extractor
dummy_extractor = BasicExtractor(node_exctractor=lambda x: x["value"].value)


# setup Evaluator
class UserEvaluator(EvalWorkChainProblem):
    problem_workchain = Sphere


class ExampleAdam(AdamOptimizer):
    evaluator_workchain = UserEvaluator
    extractor = dummy_extractor


parameters = Dict({
    "algorithm_settings": {"learning_rate": 0.01},
    "initial_parameters": List([0.1, -0.1, -0.2]),
})

__parameters = Dict(dict={
    "itmax": Int(100),
    "parameters": parameters,
})

results = run(
    ExampleAdam,
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
