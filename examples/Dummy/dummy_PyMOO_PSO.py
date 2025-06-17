from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, Int, Str

from aiida_reoptimize.base.Evaluation import EvalWorkChainProblem
from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.optimizers.PyMOO.PyMOO import PyMOO_Optimizer
from aiida_reoptimize.problems.problems import Sphere

load_profile()

# Setup extracor
dummy_extractor = BasicExtractor(node_extractor=lambda x: x["value"])


# setup Evaluator
class UserEvaluator(EvalWorkChainProblem):
    problem_workchain = Sphere


class ExamplePyMOO(PyMOO_Optimizer):
    evaluator_workchain = UserEvaluator
    extractor = dummy_extractor


parameters = Dict({
    'dimensions': 3,
    'bounds': [[-1.0, 3.0], [-5.0, 4.0], [-2.0, 1.0]],
    'algorithm_settings': {"pop_size": 10, "c1": 2.0, "c2": 2.0, "w": 0.5},
})

__parameters = Dict(dict={
    'itmax': Int(10),
    'parameters': parameters,
    'algorithm_name': Str('PSO'),
})

results = run(
    ExamplePyMOO,
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
