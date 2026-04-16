import numpy as np
from aiida.orm import List

from ..result_utils import ensure_population_has_valid_results
from .base import _GDBase


class BFGSOptimizer(_GDBase):
    """BFGS optimizer based on numerical gradient evaluation."""

    def initialize(self):
        super().initialize()
        self.initialize_step_control()
        self.ctx.inv_hessian = np.eye(len(self.ctx.parameters))
        self.ctx.gradient_prev = None
        self.ctx.parameters_prev = None

        # Line search parameters
        self.ctx.alpha = self.inputs["parameters"].get("algorithm_settings", {}).get("alpha") or 1.0
        self.ctx.beta = self.inputs["parameters"].get("algorithm_settings", {}).get("beta") or 0.5

        self.ctx.sigma = self.inputs["parameters"].get("algorithm_settings", {}).get("sigma") or 1e-4

        self.ctx.linesearch_max_iter = (
            self.inputs["parameters"].get("algorithm_settings", {}).get("linesearch_max_iter") or 20
        )

    def _reset_after_jump(self):
        """Reset BFGS state after a random jump."""

        self.ctx.inv_hessian = np.eye(len(self.ctx.parameters))
        self.ctx.gradient_prev = None
        self.ctx.parameters_prev = None

    def _line_search(self, direction):
        """
        Performs a backtracking line search to determine an appropriate step
        size along the given search direction. The method iteratively reduces
        the step size (alpha) by a factor of beta until the Armijo condition
        is satisfied, ensuring sufficient decrease in the objective function.
        """
        alpha = self.ctx.alpha
        beta = self.ctx.beta
        sigma = self.ctx.sigma
        max_iter = self.ctx.linesearch_max_iter

        params = self.ctx.parameters
        f0 = self.ctx.results[0]
        grad = self.ctx.gradient

        for _ in range(max_iter):
            self.report(f"Performing line search iteration {_ + 1}")
            trial_params = params + alpha * direction
            trial_targets = [trial_params.tolist()]
            raw_trial_results = self.run_evaluator(
                List(trial_targets),
                calculator_parameters=self.ctx.calculator_parameters,
            )
            extracted_trial_results = self.extractor(raw_trial_results["evaluation_results"])
            exit_code = ensure_population_has_valid_results(
                self,
                extracted_trial_results,
                raw_results=raw_trial_results["evaluation_results"],
                context=f"line search iteration {_ + 1}",
            )
            if exit_code is not None:
                return exit_code
            f_trial = extracted_trial_results[0]
            if f_trial <= f0 + sigma * alpha * np.dot(grad, direction):
                return alpha
            alpha *= beta
        self.report(
            f"Line search failed to find a suitable step size, using alpha={self.ctx.alpha * 1e-3}"  # noqa: E501
        )
        return self.ctx.alpha * 1e-3

    def update_parameters(self, gradient: np.ndarray):
        """Update parameters using BFGS direction and step size."""

        self.record_history(
            parameters=self.ctx.parameters,
            gradient=gradient,
            value=self.ctx.results[0],
        )

        exit_code = self.handle_worse_objective(
            rate_key="alpha",
            on_jump=self._reset_after_jump,
        )
        if exit_code is not None:
            return exit_code

        if self.ctx.iteration == 1:
            # First iteration, no previous gradient/parameters
            direction = -np.dot(self.ctx.inv_hessian, gradient)
        else:
            s = self.ctx.parameters - self.ctx.parameters_prev
            y = gradient - self.ctx.gradient_prev
            ys = np.dot(y, s)
            if ys > self.ctx.epsilon:  # Avoid division by zero
                I = np.eye(len(self.ctx.parameters))  # noqa: E741
                rho = 1.0 / ys
                V = I - rho * np.outer(s, y)
                self.ctx.inv_hessian = V @ self.ctx.inv_hessian @ V.T + rho * np.outer(s, s)
            direction = -np.dot(self.ctx.inv_hessian, gradient)

        step_size = self._line_search(direction)
        if hasattr(step_size, "status") and step_size.status != 0:
            return step_size

        self.report(f"Current step size is {step_size}")
        self.ctx.parameters_prev = self.ctx.parameters.copy()
        self.ctx.gradient_prev = gradient.copy()
        step = step_size * direction
        step = self.clamp_step(step)

        self.report_progress()
        self.ctx.parameters = self.ctx.parameters + step
        self.ctx.iteration += 1
