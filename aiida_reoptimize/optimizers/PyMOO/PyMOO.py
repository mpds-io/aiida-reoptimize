from typing import Type

import numpy as np
from aiida.engine import WorkChain
from aiida.orm import Dict, Float, Int, List, Str
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.problems.static import StaticProblem

from aiida_reoptimize.optimizers.OptimizerBase import _OptimizerBase
from aiida_reoptimize.optimizers.PyMOO.Builder import AlgorithmBuilder


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
            help="Optimization parameters including dimensions, \
                bounds, and algorithm settings.",
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
        ]
        self.ctx.algorithm_name = self.inputs.algorithm_name.value
        self.ctx.history = []

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

        best_value = None
        best_pk = None

        while self.check_itmax():
            pop = algorithm.ask()
            targets = List(list=pop.get("X").tolist())
            raw_results = self.run_evaluator(
                targets, calculator_parameters=self.ctx.calculator_parameters
            )
            results = self.extractor(raw_results["evaluation_results"])

            # Extract PKs for each result
            node_pks = [
                item["pk"] if isinstance(item, dict) and "pk" in item else None
                for item in raw_results["evaluation_results"]
            ]

            # Find best value and pk in this batch
            min_idx = int(np.argmin(results))
            min_value = results[min_idx]
            min_pk = (
                node_pks[min_idx] if node_pks[min_idx] is not None else None
            )

            # Update global best
            if best_value is None or min_value < best_value:
                best_value = min_value
                best_pk = min_pk

            # Record history for this iteration
            self.ctx.history.append({
                "iteration": self.ctx.iteration,
                "best_value": min_value,
                "best_pk": min_pk,
            })

            static = StaticProblem(problem, F=np.array(results))
            Evaluator().eval(static, pop)
            algorithm.tell(infills=pop)

            self.report(f"Iteration {self.ctx.iteration}: {targets}")
            self.ctx.iteration += 1

        self.ctx["best_position"] = algorithm.result().X
        self.ctx["best_value"] = algorithm.result().F
        self.ctx["best_node_pk"] = best_pk

    def finalize(self):
        """Finalize the optimization process and store results."""
        best_position = self.ctx["best_position"]
        best_value = self.ctx["best_value"]
        best_node_pk = self.ctx.get("best_node_pk", None)
        if hasattr(best_position, "tolist"):
            best_position = best_position.tolist()
        if hasattr(best_value, "item"):
            best_value = float(best_value.item())
        self.out("optimized_parameters", List(list=best_position).store())
        self.out("final_value", Float(best_value).store())
        self.out("history", List(self.ctx.history).store())
        if self.inputs.get_best.value:
            self.out("result_node_pk", Int(best_node_pk).store())

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
