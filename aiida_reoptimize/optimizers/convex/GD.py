import numpy as np

from .base import _GDBase


class RMSpropOptimizer(_GDBase):
    """
    RMSprop workchain
    """

    def initialize(self):
        super().initialize()
        self.ctx.accumulated_grad_sq = np.zeros_like(self.ctx.parameters)
        self.ctx.learning_rate = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("learning_rate")
            or 1e-3
        )
        self.ctx.rho = (
            self.inputs["parameters"].get("algorithm_settings", {}).get("rho")
            or 0.9
        )

    def update_parameters(self, gradient: np.array):
        """Update parameters using RMSprop algorithm."""
        self.record_history(
            parameters=self.ctx.parameters,
            gradient=gradient,
            value=self.ctx.results[0],
        )

        # ! XXX add np.inf / np.nan safety check
        self.ctx.accumulated_grad_sq = (
            self.ctx.rho * self.ctx.accumulated_grad_sq
            + (1 - self.ctx.rho) * gradient**2
        )

        step = (
            self.ctx.learning_rate
            / np.sqrt(self.ctx.accumulated_grad_sq + self.ctx.epsilon)
            * gradient
        )
        self.report_progress()
        self.ctx.parameters -= step
        self.ctx.iteration += 1


class AdamOptimizer(_GDBase):
    def initialize(self):
        super().initialize()
        self.ctx.m = np.zeros_like(self.ctx.parameters)
        self.ctx.v = np.zeros_like(self.ctx.parameters)
        self.ctx.learning_rate = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("learning_rate", 5e-2)
        )

        self.ctx.beta1 = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("beta1", 0.5)
        )

        self.ctx.beta2 = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("beta2", 0.999)
        )

    def update_parameters(self, gradient: np.array):
        """Update parameters using ADAM algorithm."""

        self.record_history(
            parameters=self.ctx.parameters,
            gradient=gradient,
            value=self.ctx.results[0],
        )

        self.ctx.m = (
            self.ctx.beta1 * self.ctx.m
            + (1 - self.ctx.beta1) * self.ctx.gradient
        )
        self.ctx.v = self.ctx.beta2 * self.ctx.v + (1 - self.ctx.beta2) * (
            gradient**2
        )

        m_hat = self.ctx.m / (1 - self.ctx.beta1**self.ctx.iteration)
        v_hat = self.ctx.v / (1 - self.ctx.beta2**self.ctx.iteration)

        denominator = np.sqrt(v_hat) + self.ctx.epsilon
        denominator[denominator <= 0] = (
            self.ctx.epsilon
        )  # must be positive

        step = self.ctx.learning_rate * m_hat / denominator

        if np.any(np.isnan(step)) or np.any(np.isinf(step)):
            self.report("Aborting: Invalid step (NaN/Inf detected).")
            self.ctx.converged = True
            return

        self.report_progress()
        self.ctx.parameters -= step
        self.ctx.iteration += 1


class ConjugateGradientOptimizer(_GDBase):
    """
    Conjugate Gradient Descent optimizer (Polak–Ribiere version) with dynamic learning rate.
    """

    def initialize(self):
        super().initialize()
        self.ctx.prev_gradient = np.zeros_like(self.ctx.parameters)
        self.ctx.direction = np.zeros_like(self.ctx.parameters)
        self.ctx.learning_rate = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("learning_rate")
            or 1e-2
        )
        self.ctx.lr_min = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("lr_min")
            or 1e-8
        )
        self.ctx.lr_max = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("lr_max")
            or 1.0
        )
        self.ctx.lr_increase = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("lr_increase")
            or 1.1
        )
        self.ctx.lr_decrease = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("lr_decrease")
            or 0.5
        )
        self.ctx.prev_value = None
        self.ctx._pending_lr_adjustment = None

    def optimization_process(self):
        """Main optimization loop for Conjugate Gradient Descent with dynamic learning rate."""
        while self.should_continue():
            targets = self.generate_targets()
            raw_results = self.run_evaluator(
                targets, calculator_parameters=self.ctx.calculator_parameters
            )
            self.ctx.raw_results = raw_results["evaluation_results"]
            self.ctx.results = self.extractor(self.ctx.raw_results)

            # Adjust learning rate and possibly revert step after new objective value is available
            self.adjust_learning_rate_after_objective()

            self.ctx.gradient = self.evaluate_gradient_numerically(
                self.ctx.results
            )
            self.update_parameters(self.ctx.gradient)

        if not self.ctx.converged:
            self.report(
                f"Optimization did not converge after {self.ctx.itmax} iterations."
            )
            return self.exit_codes.ERROR_MAX_ITERATIONS
        else:
            self.report(
                f"Optimization converged after {self.ctx.iteration} iterations."
            )

    def update_parameters(self, gradient: np.array):
        """Update parameters using Conjugate Gradient Descent (Polak–Ribiere) with dynamic learning rate."""
        self.record_history(
            parameters=self.ctx.parameters,
            gradient=gradient,
            value=self.ctx.results[0],
        )

        # CGD restart every N iterations
        restart_interval = 10
        if self.ctx.iteration == 1 or self.ctx.iteration % restart_interval == 0:
            self.report("Restarting CGD direction.")
            self.ctx.direction = -gradient
        else:
            y = gradient - self.ctx.prev_gradient
            denom = np.dot(self.ctx.prev_gradient, self.ctx.prev_gradient) + self.ctx.epsilon
            beta = np.dot(gradient, y) / denom if denom > 1e-12 else 0.0
            beta = max(beta, 0)
            self.ctx.direction = -gradient + beta * self.ctx.direction

        step = self.ctx.learning_rate * self.ctx.direction

        if np.any(np.isnan(step)) or np.any(np.isinf(step)):
            self.report("Aborting: Invalid step (NaN/Inf detected).")
            self.ctx.converged = True
            return

        # Save current value, parameters, and prev_gradient before update
        prev_value = self.ctx.results[0]
        prev_parameters = self.ctx.parameters.copy()
        prev_prev_gradient = self.ctx.prev_gradient.copy()

        self.ctx.parameters += step
        self.ctx.iteration += 1

        # Defer learning rate adjustment until after new objective value is available
        self.ctx._pending_lr_adjustment = {
            "prev_value": prev_value,
            "prev_parameters": prev_parameters,
            "prev_prev_gradient": prev_prev_gradient,
        }

        self.report(f"Iteration {self.ctx.iteration}: Learning rate = {self.ctx.learning_rate:.6f}, Step = {step}")
        self.ctx.prev_gradient = gradient.copy()
        self.report_progress()

    def adjust_learning_rate_after_objective(self):
        """Call this after the new objective value is available."""
        if not self.ctx._pending_lr_adjustment:
            return

        prev_value = self.ctx._pending_lr_adjustment["prev_value"]
        prev_parameters = self.ctx._pending_lr_adjustment["prev_parameters"]
        prev_prev_gradient = self.ctx._pending_lr_adjustment["prev_prev_gradient"]
        current_value = self.ctx.results[0]

        if prev_value is not None:
            # if results improved, increase learning rate
            if current_value < prev_value:
                self.ctx.prev_value = current_value
                self.ctx.learning_rate = min(self.ctx.learning_rate * self.ctx.lr_increase, self.ctx.lr_max)
            # else, decrease learning rate and revert step
            else:
                self.ctx.learning_rate = max(self.ctx.learning_rate * self.ctx.lr_decrease, self.ctx.lr_min)
                self.ctx.parameters = prev_parameters  # revert update
                self.ctx.prev_gradient = prev_prev_gradient.copy()  # revert prev_gradient
                self.ctx.gradient = self.ctx.prev_gradient.copy()  # revert current gradient
                self.ctx.prev_value = prev_value

        self.ctx._pending_lr_adjustment = None
