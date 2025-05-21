from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, Int

from src.base.Evaluation import EvalWorkChainProblem
from src.optimizers.PyMOO import PyMOO_PSO
from src.problems.problems import Sphere

load_profile()


# setup Evaluator
class UserEvaluator(EvalWorkChainProblem):
    problem_workchain = Sphere
    extractor = staticmethod(lambda x: x["value"].value)


class ExamplePSO(PyMOO_PSO):
    evaluator_workchain = UserEvaluator


parameters = Dict({
    'num_particles': 10,
    'dimensions': 2,
    'bounds': [[-1.0, 3.0], [-3.0, 2.0]],
})

__parameters = Dict(dict={
    'itmax': Int(25),
    'parameters': parameters
})

results = run(
    ExamplePSO,
    **__parameters,
)

print("Optimization Results:")
print(f"Best position: {results['optimized_parameters']}")
print(f"Best energy: {results['final_value']}")