import numpy as np
from aiida.engine import run
from aiida.orm import Float, List

from ..base.OptimizerBase import _OptimizerBase


class _SDGBase(_OptimizerBase):
    """Basis for SDG based optimization algorithm"""

    def initialize(self):
        """Initialize context variables and optimization parameters."""
        self.ctx.parameters = np.array(
            self.inputs["parameters"]["initial_parameters"]
        )
        self.ctx.tolerance = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("tolerance")
            or 1e-3
        )
        self.ctx.itmax = (
            self.inputs.itmax.value
        )  # +1 to include the last iteration # TODO make it everywere
        self.ctx.epsilon = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("epsilon")
            or 1e-8
        )
        self.ctx.delta = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("delta")
            or 1e-6
        )
        self.ctx.converged = False
        self.ctx.iteration = 1

    def should_continue(self):
        return not self.ctx.converged and self.ctx.iteration < self.ctx.itmax

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
        gradient = [
            (results[i + 1] - func_value) / self.ctx.delta
            for i in range(len(self.ctx.parameters))
        ]
        gradient = np.array(gradient)

        if np.linalg.norm(gradient) < self.ctx.tolerance:
            self.ctx.converged = True
        return gradient

    def update_parameters(self):
        raise NotImplementedError(
            "Subclasses must implement update_parameters()"
        )

    def optimization_process(self):
        """Main optimization loop for SDG based algorithms."""
        while self.should_continue():
            targets = self.generate_targets()
            results = run(self.evaluator_workchain, targets=targets)
            self.ctx.results = results["evaluation_results"]
            self.ctx.gradient = self.evaluate_gradient_numerically(
                self.ctx.results
            )
            self.update_parameters(self.ctx.gradient)

        if not self.ctx.converged:
            # ! TODO add error exit
            self.report(
                "Optimization did not converge within the maximum iterations."
            )

    def finalize(self):
        self.out(
            "optimized_parameters",
            List(list=self.ctx.parameters.tolist()).store(),
        )
        self.out("final_value", Float(self.ctx.results[0]).store())


class RMSpropOptimizer(_SDGBase):
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
        )  # noqa: E501
        self.ctx.rho = (
            self.inputs["parameters"].get("algorithm_settings", {}).get("rho")
            or 0.9
        )

    def update_parameters(self, gradient: np.array):
        """Update parameters using RMSprop algorithm."""
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
        self.report(
            f"\nIteration:{self.ctx.iteration}/{self.ctx.itmax},\nCurrent parameters is:{self.ctx.parameters}\nCurrent gradient norm is: {np.linalg.norm(gradient)}\nCurrent  target value is: {self.ctx.results[0]}"  # noqa: E501
        )
        self.ctx.parameters -= step


class AdamOptimizer(_SDGBase):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        # ! XXX Add information about parameters / test paramiters to find optimal  # noqa: E501
        spec.input(
            "learning_rate", valid_type=Float, default=lambda: Float(0.001)
        )
        spec.input("beta1", valid_type=Float, default=lambda: Float(0.9))
        spec.input("beta2", valid_type=Float, default=lambda: Float(0.999))

    def initialize(self):
        super().initialize()
        self.ctx.m = np.zeros_like(self.ctx.parameters)
        self.ctx.v = np.zeros_like(self.ctx.parameters)
        self.ctx.learning_rate = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("learning_rate", 1e-3)
        )

        self.ctx.beta1 = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("beta1", 0.9)
        )

        self.ctx.beta2 = (
            self.inputs["parameters"]
            .get("algorithm_settings", {})
            .get("beta2", 0.999)
        )

    def update_parameters(self, gradient: np.array):
        """Update parameters using ADAM algorithm."""

        self.ctx.m = (
            self.ctx.beta1 * self.ctx.m
            + (1 - self.ctx.beta1) * self.ctx.gradient
        )
        self.ctx.v = self.ctx.beta2 * self.ctx.v + (
            1 - self.ctx.beta2
        ) * (gradient**2)

        m_hat = self.ctx.m / (1 - self.ctx.beta1**self.ctx.iteration)
        v_hat = self.ctx.v / (1 - self.ctx.beta2**self.ctx.iteration)

        denominator = np.sqrt(v_hat) + self.ctx.epsilon
        denominator[denominator <= 0] = (
            self.ctx.epsilon
        )  # Гарантируем положительность

        step = self.ctx.learning_rate * m_hat / denominator

        if np.any(np.isnan(step)) or np.any(np.isinf(step)):
            self.report("Aborting: Invalid step (NaN/Inf detected).")
            self.ctx.converged = True
            return

        self.report(
            f"\nIteration:{self.ctx.iteration}/{self.ctx.itmax},\nCurrent parameters is:{self.ctx.parameters}\nCurrent gradient norm is: {np.linalg.norm(gradient)}\nCurrent  target value is: {self.ctx.results[0]}"  # noqa: E501
        )
        self.ctx.parameters -= step
        self.ctx.iteration += 1
