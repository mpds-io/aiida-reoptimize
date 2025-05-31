from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, Int, Str

from aiida_reoptimize.base.Evaluation import EvalWorkChainProblem
from aiida_reoptimize.optimizers.PyMOO.PyMOO import PyMOO_Optimizer
from aiida_reoptimize.problems.problems import Sphere

load_profile()


# setup Evaluator
class UserEvaluator(EvalWorkChainProblem):
    problem_workchain = Sphere
    extractor = staticmethod(lambda x: x["value"].value)


class ExamplePyMOO(PyMOO_Optimizer):
    evaluator_workchain = UserEvaluator


parameters = Dict({
    'dimensions': 3,
    'bounds': [[-1.0, 3.0], [-5.0, 4.0], [-2.0, 1.0]],
    'algorithm_settings': {"pop_size": 2, "c1": 2.0, "c2": 2.0, "w": 0.5},
})

__parameters = Dict(dict={
    'itmax': Int(2),
    'parameters': parameters,
    'algorithm_name': Str('PSO'),
})

results = run(
    ExamplePyMOO,
    **__parameters,
)

print("Optimization Results:")
print(f"Best position: {results['optimized_parameters']}")
print(f"Best value: {results['final_value']}")