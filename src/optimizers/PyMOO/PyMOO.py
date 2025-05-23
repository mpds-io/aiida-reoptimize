from typing import Type

import numpy as np
from aiida.engine import WorkChain, run
from aiida.orm import Dict, Float, Int, List, Str
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.problems.static import StaticProblem

from src.base.OptimizerBase import _OptimizerBase
from src.optimizers.PyMOO.Builder import AlgorithmBuilder


class _PyMOO_Base(_OptimizerBase):
    evaluator_workchain: Type[WorkChain]

    @classmethod
    def define(cls, spec):
        """Define the inputs and outputs of the WorkChain."""
        assert cls.evaluator_workchain is not None, "evaluator must be set"
        super().define(spec)
        spec.input("algorithm_name", valid_type=Str, help="Algorithm name.")
        spec.input(
            "parameters",
            valid_type=Dict,
            help="Optimization parameters including dimensions, bounds, and algorithm settings.",
        )
        spec.input(
            "itmax", valid_type=Int, help="Maximum number of iterations."
        )

    def initialize(self):
        """Initialize most basic parameters."""
        self.ctx.iteration = 0
        self.ctx.max_iterations = self.inputs.itmax.value
        self.ctx.dimensions = self.inputs.parameters["dimensions"]
        self.ctx.bounds = np.array(self.inputs.parameters["bounds"])
        self.ctx.algorithm_settings = self.inputs.parameters[
            "algorithm_settings"
        ]  # noqa: E501
        self.ctx.algorithm_name = self.inputs.algorithm_name.value

    def define_problem(self) -> Problem:
        """Define a PyMOO problem instance."""

        class MyProblem(Problem):
            def __init__(self, dimensions, xl, xu, **kwargs):
                super().__init__(
                    n_var=dimensions,
                    n_obj=1,
                    n_ieq_constr=0,
                    xl=xl,
                    xu=xu,
                    **kwargs,
                )

        xl, xu = self.ctx.bounds[:, 0], self.ctx.bounds[:, 1]
        dimensions = self.ctx.dimensions
        return MyProblem(dimensions=dimensions, xl=xl, xu=xu)

    def optimization_process(self):
        """Main optimization loop."""
        problem = self.define_problem()
        algorithm = self.define_algorithm(problem)

        while self.check_itmax():
            pop = algorithm.ask()
            targets = List(list=pop.get("X").tolist())
            results = run(self.evaluator_workchain, targets=targets)

            static = StaticProblem(
                problem, F=np.array(results["evaluation_results"])
            )
            Evaluator().eval(static, pop)
            algorithm.tell(infills=pop)

            self.report(f"Iteration {self.ctx.iteration}: {targets}")
            self.ctx.iteration += 1

        self.ctx["best_position"] = algorithm.result().X
        self.ctx["best_value"] = algorithm.result().F

    def finalize(self):
        """Finalize the optimization process and store results."""
        best_position = self.ctx["best_position"]
        best_value = self.ctx["best_value"]
        # Convert numpy arrays to lists/floats
        if hasattr(best_position, "tolist"):
            best_position = best_position.tolist()
        if hasattr(best_value, "item"):
            best_value = float(best_value.item())
        self.out("optimized_parameters", List(list=best_position).store())
        self.out("final_value", Float(best_value).store())

    def define_algorithm(self):
        raise NotImplementedError(
            "Subclasses must implement define_algorithm()"
        )


class PyMOO_Optimizer(_PyMOO_Base):
    def define_algorithm(self, problem):
        algorithm = AlgorithmBuilder.build_algorithm(
            self.ctx.algorithm_name, **self.ctx.algorithm_settings
        )
        algorithm.setup(problem)
        return algorithm
