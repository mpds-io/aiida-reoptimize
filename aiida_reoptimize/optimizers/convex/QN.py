import numpy as np
from aiida.orm import List

from .base import _GDBase


class BFGSOptimizer(_GDBase):
    """BFGS optimizer based on numerical gradient evaluation."""

    def initialize(self):
        super().initialize()
        self.ctx.inv_hessian = np.eye(len(self.ctx.parameters))
        self.ctx.gradient_prev = None
        self.ctx.parameters_prev = None

        # Line search parameters
        self.ctx.alpha = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("alpha")
            or 1.0
        )
        self.ctx.beta = (
            self.inputs["parameters"].get("algorithm_settings", {}).get("beta")
            or 0.5
        )

        self.ctx.sigma = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("sigma")
            or 1e-4
        )

        self.ctx.linesearch_max_iter = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("linesearch_max_iter")
            or 20
        )

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
            trial_params = params + alpha * direction
            trial_targets = [trial_params.tolist()]
            raw_trial_results = self.run_evaluator(
                List(trial_targets),
                calculator_parameters=self.ctx.calculator_parameters,
            )
            f_trial = self.extractor(raw_trial_results["evaluation_results"])[
                0
            ]
            if f_trial <= f0 + sigma * alpha * np.dot(grad, direction):
                return alpha
            alpha *= beta
        self.report(
            f"Line search failed to find a suitable step size, using alpha={self.ctx.alpha * 1e-3}"  # noqa: E501
        )
        return self.ctx.alpha * 1e-3

    def update_parameters(self, gradient: np.array):
        """Update parameters using BFGS direction and step size."""

        self.record_history(
            parameters=self.ctx.parameters,
            gradient=gradient,
            value=self.ctx.results[0],
        )

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
                self.ctx.inv_hessian = (
                    V @ self.ctx.inv_hessian @ V.T + rho * np.outer(s, s)
                )
            direction = -np.dot(self.ctx.inv_hessian, gradient)

        step_size = self._line_search(direction)

        self.report(f"Current step size is {step_size}")
        self.ctx.parameters_prev = self.ctx.parameters.copy()
        self.ctx.gradient_prev = gradient.copy()
        self.report_progress()
        self.ctx.parameters = self.ctx.parameters + step_size * direction
        self.ctx.iteration += 1
