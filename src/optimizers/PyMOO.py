from typing import Type

import numpy as np
from aiida.engine import WorkChain, run
from aiida.orm import Float, List
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

from src.base.OptimizerBase import _OptimizerBase


class _PyMOO_Base(_OptimizerBase):
    evaluator_workchain: Type[WorkChain]

    @classmethod
    def define(cls, spec):
        """Define the inputs and outputs of the WorkChain."""
        assert cls.evaluator_workchain is not None, "evalutor must be set"
        super().define(spec)

    def initialize(self):
        """Initialize PSO parameters and particle positions."""
        self.ctx.num_particles = self.inputs.parameters["num_particles"]
        self.ctx.iteration = 0
        self.ctx.max_iterations = self.inputs.itmax.value
        self.ctx.dimensions = self.inputs.parameters["dimensions"]
        self.ctx.bounds = np.array(self.inputs.parameters["bounds"])
        # self.ctx.penalty = self.inputs.penalty.value

    def define_problem(self) -> Problem:
        """Define a PyMOO problem instance."""
        xl, xu = self.ctx.bounds[:, 0], self.ctx.bounds[:, 1]
        dimensions = self.ctx.dimensions

        class MyProblem(Problem):
            def __init__(self, **kwargs):
                super().__init__(
                    n_var=dimensions,
                    n_obj=1,
                    n_ieq_constr=0,
                    xl=xl,
                    xu=xu,
                    **kwargs,
                )

        return MyProblem()

    def optimization_process(self):
        """Main optimization loop."""
        problem = self.define_problem()
        algorithm = self.define_algorithm(problem)

        while self.check_itmax():
            pop = algorithm.ask()
            targets = List(pop.get("X").tolist())
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
            "Subclasses must implement define_algotithm()"
        )


class PyMOO_PSO(_PyMOO_Base):
    """
    PyMOO WorkChain for CMA-ES optimization.

    Inherits from _PyMOO_Base and implements the CMA-ES algorithm.
    """

    def define_algorithm(self, problem):
        algorithm = PSO(
            pop_size=self.ctx.num_particles,
            termination=NoTermination(),
        )
        algorithm.setup(problem)
        return algorithm
