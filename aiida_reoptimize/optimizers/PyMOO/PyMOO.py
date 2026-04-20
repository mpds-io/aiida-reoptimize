from typing import Type

import numpy as np
from aiida.engine import WorkChain
from aiida.orm import Dict, Float, Int, List, Str
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.problems.static import StaticProblem

from aiida_reoptimize.optimizers.OptimizerBase import _OptimizerBase
from aiida_reoptimize.optimizers.parameter_utils import prepare_optimization_parameters
from aiida_reoptimize.optimizers.PyMOO.Builder import AlgorithmBuilder
from aiida_reoptimize.optimizers.result_utils import ensure_population_has_valid_results


class _PyMOO_Base(_OptimizerBase):
    """Base class for PyMOO-backed optimization WorkChains.

    Manages the ask-evaluate-tell loop with AiiDA, where each iteration
    submits a batch of evaluations via the evaluator workchain and feeds
    the results back to the PyMOO algorithm. Supports early stopping via
    ``tol`` (spread of best values over 3 iterations).

    Subclasses must set ``evaluator_workchain`` and implement
    ``define_algorithm``.
    """

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
            help="Optimization parameters including bounds, optional tol, and algorithm settings.",
        )
        spec.input("itmax", valid_type=Int, help="Maximum number of iterations.")
        spec.input("itmin", valid_type=Int, default=lambda: Int(10), help="Maximum number of iterations.")
        spec.exit_code(
            401,
            "ERROR_NO_VALID_SOLUTION",
            message="Optimization failed to find a valid solution.",
        )

    def initialize(self):
        """Initialize most basic parameters."""
        super().initialize()

        parameters_dict = self.inputs.parameters.get_dict()
        normalized = prepare_optimization_parameters(
            parameters_dict,
            structure=self.inputs.get("structure"),
            require_bounds=True,
            require_initial_parameters=False,
        )

        self.ctx.iteration = 0
        self.ctx.max_iterations = self.inputs.itmax.value
        self.ctx.min_iterations = self.inputs.itmin.value
        self.ctx.dimensions = normalized["dimensions"]
        self.ctx.bounds = normalized["bounds"]
        algorithm_settings = dict(parameters_dict.get("algorithm_settings", {}))
        tol = parameters_dict.get("tol", algorithm_settings.pop("tol", None))
        if tol is not None:
            tol = float(tol)
            if not np.isfinite(tol) or tol < 0:
                raise ValueError("'tol' must be a finite non-negative number.")
        self.ctx.tol = tol
        self.ctx.algorithm_settings = algorithm_settings
        calculator_parameters = parameters_dict.get("calculator_parameters")
        self.ctx.calculator_parameters = Dict(dict=calculator_parameters) if calculator_parameters is not None else None

        self.ctx.algorithm_name = self.inputs.algorithm_name.value
        self.ctx.terminated_by_tol = False

    def report_progress(self):
        if not self.ctx.history:
            return
        entry = self.ctx.history[-1]
        best_pos = entry.get("best_position")
        positions = entry.get("positions")

        parts = [
            f"Iteration {entry['iteration']}/{self.ctx.max_iterations}",
            f"best_value={entry['value']:.6e}",
            f"best_position={best_pos}" if best_pos is not None else "best_position=N/A",
        ]
        if positions is not None:
            parts.append(f"population_size={len(positions)}")
        if entry.get("result_node_pk") is not None:
            parts.append(f"pk={entry['result_node_pk']}")

        self.report(" | ".join(parts))

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
        best_position = None
        recent_best_values = []

        while self.check_itmax():
            pop = algorithm.ask()
            pop_x = np.array(pop.get("X"), dtype=np.float64)
            targets = List(list=pop.get("X").tolist())
            run_kwargs = {}
            if self.ctx.calculator_parameters is not None:
                run_kwargs["calculator_parameters"] = self.ctx.calculator_parameters
            raw_results = self.run_evaluator(targets, **run_kwargs)
            results = self.extractor(raw_results["evaluation_results"])
            exit_code = ensure_population_has_valid_results(
                self,
                results,
                raw_results=raw_results["evaluation_results"],
                context=f"iteration {self.ctx.iteration}",
            )
            if exit_code is not None:
                return exit_code

            # Extract PKs for each result
            node_pks = [
                item["pk"] if isinstance(item, dict) and "pk" in item else None
                for item in raw_results["evaluation_results"]
            ]

            # Find best value and pk in this batch
            min_idx = int(np.argmin(results))
            min_value = float(results[min_idx])
            min_pk = node_pks[min_idx] if node_pks[min_idx] is not None else None
            min_position = pop_x[min_idx].copy()

            # Update global best
            if best_value is None or min_value < best_value:
                best_value = min_value
                best_pk = min_pk
                best_position = min_position

            recent_best_values.append(min_value)
            if len(recent_best_values) > 3:
                recent_best_values = recent_best_values[-3:]

            entry = self.record_history(
                iteration=self.ctx.iteration,
                value=min_value,
                result_node_pk=min_pk,
            )
            entry["best_position"] = min_position.tolist()
            entry["positions"] = pop_x.tolist()

            static = StaticProblem(problem, F=np.array(results))
            Evaluator().eval(static, pop)
            algorithm.tell(infills=pop)

            if self.ctx.tol is not None and len(recent_best_values) == 3:
                spread = max(recent_best_values) - min(recent_best_values)
                if spread < self.ctx.tol and self.ctx.iteration > self.ctx.min_iterations:
                    self.ctx.terminated_by_tol = True
                    self.report(
                        f"Stopping early: spread of last 3 best values ({spread:.6e}) is below tol={self.ctx.tol:.6e}."
                    )
                    self.ctx.iteration += 1
                    break

            self.report_progress()
            self.ctx.iteration += 1

        if best_position is None or best_value is None:
            result = algorithm.result()
            self.ctx["best_position"] = result.X
            self.ctx["best_value"] = result.F
        else:
            self.ctx["best_position"] = best_position
            self.ctx["best_value"] = best_value
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
        if best_value == self.extractor.get_penalty():
            return self.exit_codes.ERROR_NO_VALID_SOLUTION
        self.out("optimized_parameters", List(list=best_position).store())
        self.out("final_value", Float(best_value).store())
        self.out("history", List(self.ctx.history).store())
        if self.inputs.get_best.value:
            self.out("result_node_pk", Int(best_node_pk).store())

    def define_algorithm(self, problem):
        raise NotImplementedError("Subclasses must implement define_algorithm()")


class PyMOO_Optimizer(_PyMOO_Base):
    """PyMOO optimizer WorkChain that delegates algorithm construction to ``AlgorithmBuilder``.

    The algorithm name and settings are provided via the ``algorithm_name``
    and ``parameters.algorithm_settings`` inputs.
    """

    def define_algorithm(self, problem):
        """Build and set up a PyMOO algorithm from ``self.ctx.algorithm_name``."""
        algorithm = AlgorithmBuilder.build_algorithm(self.ctx.algorithm_name, **self.ctx.algorithm_settings)
        algorithm.setup(problem)
        return algorithm
