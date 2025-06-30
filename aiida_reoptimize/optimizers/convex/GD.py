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
