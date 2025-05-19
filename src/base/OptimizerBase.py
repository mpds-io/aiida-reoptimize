from typing import Type

from aiida.engine import WorkChain
from aiida.orm import Dict, Float, Int, List, Str


class _OptimizerBase(WorkChain):
    """Base class for optimization algorithms."""

    evaluator_workchain: Type[WorkChain]

    @classmethod
    def define(cls, spec):
        assert cls.evaluator_workchain is not None, "evaluator must be set"  # noqa: E501
        super().define(spec)
        spec.input(
            "parameters", valid_type=Dict, help="Optimization parameters."
        )
        spec.input(
            "itmax",
            valid_type=Int,
            default=lambda: Int(100),
            help="Maximum number of iterations.",
        )
        spec.input(
            "key_value",
            valid_type=Str,
            default=lambda: Str("energy"),
            help="Key for the target value.",
        )

        spec.outline(cls.initialize, cls.optimization_process, cls.finalize)

        spec.output(
            "optimized_parameters",
            valid_type=Dict,
            required=True,
            help="Optimized parameters.",
        )
        spec.output(
            "final_value",
            valid_type=Float,
            required=True,
            help="Final value of the objective function.",
        )

    def initialize(self):
        raise NotImplementedError("Subclasses must implement initialize()")

    def optimization_process(self):
        """Main optimization loop."""
        raise NotImplementedError(
            "Subclasses must implement optimization_process()"
        )

    def finalize(self):
        """Finalize the optimization process."""
        self.out(
            "optimized_parameters",
            List(list=self.ctx.best_parameters.tolist()).store(),
        )
        self.out("final_value", Float(self.ctx.results[0]).store())

    def check_itmax(self):
        """Check if the optimization should continue."""
        return self.ctx.iteration < self.inputs.itmax.value
