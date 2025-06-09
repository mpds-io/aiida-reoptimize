from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, Int, List

from aiida_reoptimize.base.Evaluation import EvalWorkChainProblem
from aiida_reoptimize.optimizers.convex.GD import RMSpropOptimizer
from aiida_reoptimize.problems.problems import Sphere

load_profile()


# setup Evaluator
class UserEvaluator(EvalWorkChainProblem):
    problem_workchain = Sphere
    extractor = staticmethod(lambda x: x["value"].value)


class ExampleRMSprop(RMSpropOptimizer):
    evaluator_workchain = UserEvaluator


parameters = Dict({
    "algorithm_settings": {"learning_rate": 0.01},
    "initial_parameters": List([0.1, -0.1, -0.2]),
})

__parameters = Dict(dict={
    "itmax": Int(35),
    "parameters": parameters,
})

results = run(
    ExampleRMSprop,
    **__parameters,
)

print("Optimization Results:")
if results:
    print(f"Best position: {results['optimized_parameters']}")
    print(f"Best value: {results['final_value']}")
    print(f"History: {results['history']}")
