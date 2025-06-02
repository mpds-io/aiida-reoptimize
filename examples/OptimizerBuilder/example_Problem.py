from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, Int, List

from aiida_reoptimize.base.OptimizerBuilder import OptimizerBuilder
from aiida_reoptimize.optimizers.convex.QN import BFGSOptimizer
from aiida_reoptimize.problems.problems import Sphere

load_profile()

builder = OptimizerBuilder.from_problem(
    optimizer_workchain=BFGSOptimizer,
    problem_workchain=Sphere,
    extractor=lambda x: x["value"].value,
)

optimizer_parameters = {
    "itmax": Int(20),
    "parameters": Dict({
        "algorithm_settings": {"tolerance": 1e-8},
        "initial_parameters": List([0.1, -0.3, 0.7]),
    })
}

optimizer = builder.get_optimizer()
results = run(optimizer, **optimizer_parameters)