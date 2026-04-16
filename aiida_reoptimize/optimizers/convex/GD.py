import numpy as np

from .base import _GDBase


class RMSpropOptimizer(_GDBase):
    """
    RMSprop workchain
    """

    def initialize(self):
        super().initialize()
        self.initialize_step_control()
        self.ctx.accumulated_grad_sq = np.zeros_like(self.ctx.parameters)
        self.ctx.learning_rate = self.inputs["parameters"].get("algorithm_settings", {}).get("learning_rate") or 1e-3
        self.ctx.rho = self.inputs["parameters"].get("algorithm_settings", {}).get("rho") or 0.9

    def _reset_after_jump(self):
        """Reset RMSprop accumulators after a random jump."""

        self.ctx.accumulated_grad_sq = np.zeros_like(self.ctx.parameters)

    def update_parameters(self, gradient: np.ndarray):
        """Update parameters using RMSprop algorithm."""
        exit_code = self.handle_worse_objective(
            rate_key="learning_rate",
            on_jump=self._reset_after_jump,
        )
        if exit_code is not None:
            return exit_code

        self.ctx.accumulated_grad_sq = self.ctx.rho * self.ctx.accumulated_grad_sq + (1 - self.ctx.rho) * gradient**2

        step = self.ctx.learning_rate / np.sqrt(self.ctx.accumulated_grad_sq + self.ctx.epsilon) * gradient

        step = self.clamp_step(step)

        if np.any(np.isnan(step)) or np.any(np.isinf(step)):
            self.report("Aborting: Invalid step (NaN/Inf detected).")
            self.ctx.converged = True
            return

        self.record_history(
            parameters=self.ctx.parameters,
            gradient=gradient,
            value=self.ctx.results[0],
            step_rate=self.ctx.learning_rate,
            step=step,
        )

        self.report_progress()
        self.ctx.parameters -= step
        self.ctx.iteration += 1


class AdamOptimizer(_GDBase):
    def initialize(self):
        super().initialize()
        self.initialize_step_control()
        self.ctx.m = np.zeros_like(self.ctx.parameters)
        self.ctx.v = np.zeros_like(self.ctx.parameters)
        self.ctx.learning_rate = self.inputs["parameters"].get("algorithm_settings", {}).get("learning_rate", 5e-2)

        self.ctx.beta1 = self.inputs["parameters"].get("algorithm_settings", {}).get("beta1", 0.5)

        self.ctx.beta2 = self.inputs["parameters"].get("algorithm_settings", {}).get("beta2", 0.999)

    def _reset_after_jump(self):
        """Reset Adam moments after a random jump."""

        self.ctx.m = np.zeros_like(self.ctx.parameters)
        self.ctx.v = np.zeros_like(self.ctx.parameters)

    def update_parameters(self, gradient: np.ndarray):
        """Update parameters using ADAM algorithm."""
        exit_code = self.handle_worse_objective(
            rate_key="learning_rate",
            on_jump=self._reset_after_jump,
        )
        if exit_code is not None:
            return exit_code

        self.ctx.m = self.ctx.beta1 * self.ctx.m + (1 - self.ctx.beta1) * self.ctx.gradient
        self.ctx.v = self.ctx.beta2 * self.ctx.v + (1 - self.ctx.beta2) * (gradient**2)

        m_hat = self.ctx.m / (1 - self.ctx.beta1**self.ctx.iteration)
        v_hat = self.ctx.v / (1 - self.ctx.beta2**self.ctx.iteration)

        denominator = np.sqrt(v_hat) + self.ctx.epsilon
        denominator[denominator <= 0] = self.ctx.epsilon  # must be positive

        step = self.ctx.learning_rate * m_hat / denominator
        step = self.clamp_step(step)

        if np.any(np.isnan(step)) or np.any(np.isinf(step)):
            self.report("Aborting: Invalid step (NaN/Inf detected).")
            self.ctx.converged = True
            return

        self.record_history(
            parameters=self.ctx.parameters,
            gradient=gradient,
            value=self.ctx.results[0],
            step_rate=self.ctx.learning_rate,
            step=step,
        )

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
        self.ctx.prev_value = None
        self.ctx.stuck_counter = 0

        self.ctx.learning_rate = self.inputs["parameters"].get("algorithm_settings", {}).get("learning_rate") or 1e-2
        self.ctx.lr_min = self.inputs["parameters"].get("algorithm_settings", {}).get("lr_min") or 1e-8
        self.ctx.lr_max = self.inputs["parameters"].get("algorithm_settings", {}).get("lr_max") or 1.0
        self.ctx.lr_increase = self.inputs["parameters"].get("algorithm_settings", {}).get("lr_increase") or 1.1
        self.ctx.lr_decrease = self.inputs["parameters"].get("algorithm_settings", {}).get("lr_decrease") or 0.5
        self.ctx.restart_interval = (
            self.inputs["parameters"].get("algorithm_settings", {}).get("restart_interval") or 10
        )

        self.ctx.allow_jumps = self.inputs["parameters"].get("algorithm_settings", {}).get("allowing_jumps", True)
        self.ctx.allowed_stuck = self.inputs["parameters"].get("algorithm_settings", {}).get("allowed_stuck", 3)

    def update_parameters(self, gradient: np.ndarray):
        """Update parameters using Conjugate Gradient Descent (Polak–Ribiere) with dynamic learning rate."""
        if self.ctx.prev_value is None:
            self.ctx.prev_value = self.ctx.results[0]
            self.ctx.prev_parameters = self.ctx.parameters.copy()
            self.ctx.prev_gradient = gradient.copy()
            self.ctx.direction = -gradient
            self.report("Starting CGD direction.")
        else:
            if self.ctx.results[0] > self.ctx.prev_value:
                self.report("Reversing CGD step.")
                self.ctx.parameters = self.ctx.prev_parameters.copy()
                self.ctx.learning_rate = max(
                    self.ctx.learning_rate * self.ctx.lr_decrease,
                    self.ctx.lr_min,
                )
                self.ctx.stuck_counter += 1
                if self.ctx.stuck_counter >= self.ctx.allowed_stuck:
                    self.report("Stuck for too long, restarting CGD direction.")
                    self.ctx.direction = -gradient
                if self.ctx.stuck_counter >= (self.ctx.allowed_stuck + 1):
                    if self.ctx.allow_jumps:
                        self.report("Jump in random direction.")
                        self.ctx.stuck_counter = 0
                        self.ctx.parameters += np.random.uniform(-0.1, 0.1, size=self.ctx.parameters.shape)
                        self.ctx.prev_value = None
                        self.ctx.prev_gradient = np.zeros_like(self.ctx.parameters)
                    else:
                        self.report("Aborting: Too many stuck iterations without allowing jumps.")
                        return self.exit_codes.ERROR_STUCK_FOR_TOO_LONG
                if self.ctx.learning_rate == self.ctx.lr_min:
                    self.report("Aborting: Learning rate reached minimum.")
                    return self.exit_codes.ERROR_STUCK_FOR_TOO_LONG
            else:
                self.ctx.stuck_counter = 0
                self.ctx.learning_rate = min(
                    self.ctx.learning_rate * self.ctx.lr_increase,
                    self.ctx.lr_max,
                )
                self.ctx.prev_value = self.ctx.results[0]
                self.ctx.prev_parameters = self.ctx.parameters.copy()

                if self.ctx.iteration % self.ctx.restart_interval == 0:
                    self.report("Restarting CGD direction.")
                    self.ctx.direction = -gradient
                else:
                    y = gradient - self.ctx.prev_gradient
                    denom = np.dot(self.ctx.prev_gradient, self.ctx.prev_gradient) + self.ctx.epsilon
                    beta = np.dot(gradient, y) / denom if denom > 1e-12 else 0.0
                    beta = max(beta, 0)
                    self.ctx.direction = -gradient + beta * self.ctx.direction

                self.ctx.prev_gradient = gradient.copy()

        step = self.ctx.learning_rate * self.ctx.direction

        if np.any(np.isnan(step)) or np.any(np.isinf(step)):
            self.report("Aborting: Invalid step (NaN/Inf detected).")
            self.ctx.converged = True
            return

        step = self.clamp_step(step)

        self.record_history(
            parameters=self.ctx.parameters,
            gradient=gradient,
            value=self.ctx.results[0],
            step_rate=self.ctx.learning_rate,
            step=step,
        )

        self.ctx.parameters += step
        self.ctx.iteration += 1
        self.report_progress()
