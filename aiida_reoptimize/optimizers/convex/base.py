import numpy as np
from aiida.engine import run
from aiida.orm import Float, List

from ..OptimizerBase import _OptimizerBase


class _GDBase(_OptimizerBase):
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
        )
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