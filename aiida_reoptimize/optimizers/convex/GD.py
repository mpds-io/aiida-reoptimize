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
            or 1e-1
        )
        self.ctx.lr_min = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("lr_min")
            or 1e-6
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
            or 1.05
        )
        self.ctx.lr_decrease = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("lr_decrease")
            or 0.5
        )
        self.ctx.prev_value = None

    def update_parameters(self, gradient: np.array):
        """Update parameters using Conjugate Gradient Descent (Polak–Ribiere) with dynamic learning rate."""
        self.record_history(
            parameters=self.ctx.parameters,
            gradient=gradient,
            value=self.ctx.results[0],
        )

        if self.ctx.iteration == 1:
            self.ctx.direction = gradient
        else:
            y = gradient - self.ctx.prev_gradient
            beta = np.dot(gradient, y) / (np.dot(self.ctx.prev_gradient, self.ctx.prev_gradient) + self.ctx.epsilon)
            beta = max(beta, 0)
            self.ctx.direction = gradient - beta * self.ctx.direction

        step = self.ctx.learning_rate * self.ctx.direction

        if np.any(np.isnan(step)) or np.any(np.isinf(step)):
            self.report("Aborting: Invalid step (NaN/Inf detected).")
            self.ctx.converged = True
            return

        # Save current value before update
        current_value = self.ctx.results[0]
        prev_parameters = self.ctx.parameters.copy()

        self.ctx.parameters -= step
        self.ctx.iteration += 1

        # After update, check new value and adjust learning rate
        # (Assume self.ctx.results[0] will be updated externally after this call)
        if self.ctx.prev_value is not None:
            if current_value < self.ctx.prev_value:
                # if target value is improvemed: increase learning rate
                self.ctx.learning_rate = min(self.ctx.learning_rate * self.ctx.lr_increase, self.ctx.lr_max)
            else:
                # If no improvement: decrease learning rate and revert step
                self.ctx.learning_rate = max(self.ctx.learning_rate * self.ctx.lr_decrease, self.ctx.lr_min)
                self.ctx.parameters = prev_parameters  # revert update
        
        self.report(f"Iteration {self.ctx.iteration}: Learning rate = {self.ctx.learning_rate:.6f}, Step = {step}")
        self.ctx.prev_value = current_value
        self.ctx.prev_gradient = gradient.copy()
        self.report_progress()
