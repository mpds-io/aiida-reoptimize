import numpy as np
from aiida import load_profile
from aiida.engine import WorkChain, run
from aiida.orm import Dict, Int, List

from aiida_reoptimize.base.Evaluation import EvalWorkChainProblem
from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.optimizers.convex.QN import BFGSOptimizer

load_profile()


# This example is mented to show you have to define problem yourself
class UserProblem(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("x", valid_type=List)
        spec.outline(cls.run_calc, cls.finalize)
        spec.output("value", valid_type=Dict)

    def run_calc(self):
        x = np.array(self.inputs.x.get_list())
        self.ctx.result = [np.sum(x**2), np.sum(x**3), np.cos(x)**2]

    def finalize(self):
        data_dict = {
            "SquareSum": self.ctx.result[0],
            "CubeSum": self.ctx.result[1],
            "CosSum": self.ctx.result[2],
        }
        self.out("value", Dict(data_dict).store())


class UserEvaluator(EvalWorkChainProblem):
    problem_workchain = UserProblem


dummy_extractor = BasicExtractor(node_extractor=lambda x: x["value"]["SquareSum"])  # noqa: E501


class ExampleBFGS(BFGSOptimizer):
    evaluator_workchain = UserEvaluator
    extractor = dummy_extractor


parameters = Dict({
    "algorithm_settings": {"alpha": 0.9, "beta": 0.8},
    "initial_parameters": List([0.1, -0.1, -0.2]),
})

__parameters = Dict(
    dict={
        "itmax": Int(20),
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