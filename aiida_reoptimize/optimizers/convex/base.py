from collections.abc import Callable

import numpy as np
from aiida.orm import Float, Int, List

from ..OptimizerBase import _OptimizerBase
from ..parameter_utils import OptimizationParameterError, prepare_optimization_parameters
from ..result_utils import ensure_population_has_valid_results


class _GDBase(_OptimizerBase):
    """Base class for gradient-descent-style optimization algorithms.

    Implements numerical gradient evaluation via finite differences, step
    clamping, rollback on worse objective, learning rate decay, and random
    jump escapes. Subclasses must implement ``update_parameters``.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.exit_code(
            400,
            "ERROR_MAX_ITERATIONS",
            message="Optimization did not converge within the maximum iterations.",  # noqa: E501
        )

        spec.exit_code(
            401,
            "ERROR_NO_VALID_SOLUTION",
            message="Optimization failed to find a valid solution.",
        )

        spec.exit_code(
            402,
            "ERROR_STUCK_FOR_TOO_LONG",
            message="Optimizer stuck: step rate reached minimum or too many consecutive worse objectives.",
        )

    def initialize(self):
        """Initialize context variables and optimization parameters."""
        super().initialize()

        parameters_dict = self.inputs.parameters.get_dict()
        try:
            normalized = prepare_optimization_parameters(
                parameters_dict,
                structure=self.inputs.get("structure"),
                require_bounds=False,
                require_initial_parameters=True,
                reporter=self.report,
            )
        except OptimizationParameterError as exc:
            self.report(f"Invalid optimization parameters: {exc}")
            return self.exit_codes.ERROR_INVALID_OPTIMIZATION_PARAMETERS
        self.ctx.parameters = normalized["initial_parameters"].copy()

        self.ctx.calculator_parameters = parameters_dict.get("calculator_parameters", {})

        settings = parameters_dict.get("algorithm_settings", {})
        self.ctx.tolerance = settings.get("tolerance", 1e-3)
        self.ctx.itmax = self.inputs.itmax.value
        self.ctx.epsilon = settings.get("epsilon", 1e-7)
        self.ctx.delta = settings.get("delta", 1e-6)
        self.ctx.converged = False
        self.ctx.iteration = 1

        self.ctx.max_step = settings.get("max_step", 0.1)

    def initialize_step_control(self):
        """Initialize shared controls for step rollback and backoff."""

        settings = self.inputs.parameters.get_dict().get("algorithm_settings", {})
        self.ctx.prev_value = None
        self.ctx.prev_parameters = self.ctx.parameters.copy()
        self.ctx.stuck_counter = 0
        self.ctx.allowed_stuck = settings.get("allowed_stuck", 3)
        self.ctx.allow_jumps = settings.get("allowing_jumps", True)
        self.ctx.jump_scale = settings.get("jump_scale", 0.1)
        self.ctx.step_decrease = settings.get("lr_decrease", 0.5)
        self.ctx.min_step_rate = settings.get("lr_min", 1e-8)

    def clamp_step(self, step):
        """Clamp component-wise step size to avoid unstable updates."""

        if self.ctx.max_step:
            clamped = np.clip(step, -self.ctx.max_step, self.ctx.max_step)
            if not np.allclose(clamped, step):
                self.report(f"Step was clamped to +/-{self.ctx.max_step} to prevent instability.")
            return clamped
        return step

    def handle_worse_objective(self, rate_key: str, on_jump: Callable[[], None] | None = None):
        """Rollback parameters and reduce step rate when objective worsens."""

        current_value = self.ctx.results[0]

        if self.ctx.prev_value is None:
            self.ctx.prev_value = current_value
            self.ctx.prev_parameters = self.ctx.parameters.copy()
            self.ctx.stuck_counter = 0
            return None

        if current_value <= self.ctx.prev_value:
            self.ctx.prev_value = current_value
            self.ctx.prev_parameters = self.ctx.parameters.copy()
            self.ctx.stuck_counter = 0
            return None

        self.report("Objective increased, reversing to previous parameters.")
        self.ctx.parameters = self.ctx.prev_parameters.copy()
        self.ctx.stuck_counter += 1

        if self.ctx.stuck_counter >= self.ctx.allowed_stuck:
            self.report("Stuck for too long, reducing step rate and preparing restart.")

        if hasattr(self.ctx, rate_key):
            current_rate = getattr(self.ctx, rate_key)
            new_rate = max(
                current_rate * self.ctx.step_decrease,
                self.ctx.min_step_rate,
            )
            setattr(self.ctx, rate_key, new_rate)
            self.report(f"Reduced {rate_key} from {current_rate} to {new_rate}.")

            if np.isclose(new_rate, self.ctx.min_step_rate) and not self.ctx.allow_jumps:
                self.report(f"Aborting: {rate_key} reached minimum ({self.ctx.min_step_rate}).")
                return self.exit_codes.ERROR_STUCK_FOR_TOO_LONG

        if self.ctx.stuck_counter >= (self.ctx.allowed_stuck + 1):
            if self.ctx.allow_jumps:
                self.report("Jump in random direction to escape local stagnation.")
                self.ctx.parameters += np.random.uniform(
                    -self.ctx.jump_scale,
                    self.ctx.jump_scale,
                    size=self.ctx.parameters.shape,
                )
                self.ctx.stuck_counter = 0
                self.ctx.prev_value = None
                self.ctx.prev_parameters = self.ctx.parameters.copy()
                if on_jump is not None:
                    on_jump()
            else:
                self.report("Aborting: Too many stuck iterations without allowing jumps.")
                return self.exit_codes.ERROR_STUCK_FOR_TOO_LONG

        return None

    def should_continue(self):
        return not self.ctx.converged and self.ctx.iteration <= self.ctx.itmax

    def generate_targets(self):
        """Generate targets for numerical gradient evaluation."""
        params = self.ctx.parameters.tolist()
        targets = [params]
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += self.ctx.delta
            targets.append(params_plus)
        return List(targets)

    def evaluate_gradient_numerically(self, results):
        """Evaluate the gradient numerically using finite differences."""
        func_value = results[0]
        gradient = [(results[i + 1] - func_value) / self.ctx.delta for i in range(len(self.ctx.parameters))]
        gradient = np.array(gradient)

        if np.linalg.norm(gradient) < self.ctx.tolerance:
            self.ctx.converged = True
        return gradient

    def record_history(
        self, parameters=None, gradient=None, value=None, result_node_pk=None, step_rate=None, step=None
    ):
        """Record the current state in the optimization history."""
        entry = super().record_history(
            iteration=self.ctx.iteration,
            value=value if value is not None else self.ctx.results[0],
            result_node_pk=result_node_pk if result_node_pk is not None else self.ctx.raw_results[0]["pk"],
        )
        entry["parameters"] = parameters.tolist() if parameters is not None else self.ctx.parameters.tolist()
        entry["gradient_norm"] = (
            float(np.linalg.norm(gradient)) if gradient is not None else getattr(self.ctx, "gradient", None)
        )
        if step_rate is not None:
            entry["step_rate"] = float(step_rate)
        if step is not None:
            entry["step"] = step.tolist() if isinstance(step, np.ndarray) else step

    def update_parameters(self, gradient: np.ndarray):
        raise NotImplementedError("Subclasses must implement update_parameters()")

    def optimization_process(self):
        """Main optimization loop for SDG based algorithms."""
        while self.should_continue():
            targets = self.generate_targets()
            raw_results = self.run_evaluator(targets, calculator_parameters=self.ctx.calculator_parameters)
            if raw_results is None:
                return self.exit_codes.ERROR_EVALUATOR_FAILED
            self.ctx.raw_results = raw_results["evaluation_results"]
            self.ctx.results = self.extractor(self.ctx.raw_results)
            exit_code = ensure_population_has_valid_results(
                self,
                self.ctx.results,
                raw_results=self.ctx.raw_results,
                context=f"iteration {self.ctx.iteration}",
            )
            if exit_code is not None:
                return exit_code
            self.ctx.gradient = self.evaluate_gradient_numerically(self.ctx.results)
            exit_code = self.update_parameters(self.ctx.gradient)
            if exit_code is not None:
                return exit_code

        if not self.ctx.converged:
            self.report(
                f"Optimization did not converge after {self.ctx.itmax} iterations."  # noqa: E501
            )
            return self.exit_codes.ERROR_MAX_ITERATIONS
        else:
            self.report(
                f"Optimization converged after {self.ctx.iteration} iterations."  # noqa: E501
            )

    def report_progress(self):
        """Report the current progress of the optimization."""
        if not self.ctx.history:
            return
        entry = self.ctx.history[-1]
        grad_norm = entry.get("gradient_norm")
        step_rate = entry.get("step_rate")
        step = entry.get("step")
        step_norm = np.linalg.norm(step) if step is not None else None

        parts = [
            f"Iteration {entry['iteration']}/{self.ctx.itmax}",
            f"params={self.ctx.parameters}",
            f"grad_norm={grad_norm:.6e}" if grad_norm is not None else "grad_norm=N/A",
            f"value={entry['value']:.6e}",
        ]
        if step_rate is not None:
            parts.append(f"step_rate={step_rate:.6e}")
        if step_norm is not None:
            parts.append(f"step_norm={step_norm:.6e}")
        if entry.get("result_node_pk") is not None:
            parts.append(f"pk={entry['result_node_pk']}")

        self.report("\n".join(parts))

    def finalize(self):
        if self.ctx.results[0] == self.extractor.get_penalty():
            return self.exit_codes.ERROR_NO_VALID_SOLUTION

        self.out(
            "optimized_parameters",
            List(list=self.ctx.parameters.tolist()).store(),
        )
        self.out("final_value", Float(self.ctx.results[0]).store())
        self.out("history", List(list=self.ctx.history).store())
        if self.inputs.get_best.value:
            self.out("result_node_pk", Int(self.ctx.raw_results[0]["pk"]).store())
