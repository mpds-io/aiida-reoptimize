from typing import Type

from aiida.engine import WorkChain, run
from aiida.orm import Bool, Dict, Float, Int, List, StructureData


class _OptimizerBase(WorkChain):
    """Base class for optimization algorithms."""

    evaluator_workchain: Type[WorkChain]
    extractor: Type[callable]

    @classmethod
    def define(cls, spec):
        assert cls.evaluator_workchain is not None, "evaluator must be set"  # noqa: E501
        assert cls.extractor is not None, "extractor must be set"

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
            "get_best",
            valid_type=Bool,
            default=lambda: Bool(True),
            help="Whether to return the best result node identifier.",
        )
        
        spec.input(
            "structure",
            valid_type=StructureData,
            required=False,
            help="Chemical structure for the optimization.",
        )

        spec.outline(cls.initialize, cls.optimization_process, cls.finalize)

        spec.output(
            "optimized_parameters",
            valid_type=List,
            required=True,
            help="Optimized parameters.",
        )
        spec.output(
            "final_value",
            valid_type=Float,
            required=True,
            help="Final value of the objective function.",
        )

        spec.output(
            "history",
            valid_type=List,
            required=False,
            help="Optimization history.",
        )

        spec.output(
            "result_node_pk",
            valid_type=Int,
            required=False,
            help="Primary key of the best result node.",
        )
        # TODO add exit codes

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

        if self.inputs.get_best.value:
            self.out(
                "result_node_pk", Int(self.ctx.best_result_node_pk).store()
            )

    def run_evaluator(self, targets, **kwargs):
        """Run the evaluator workchain with or without structure input."""
        if self.inputs.get("structure"):
            return run(
                self.evaluator_workchain,
                targets=targets,
                structure=self.inputs.structure,
                **kwargs,
            )
        else:
            return run(self.evaluator_workchain, targets=targets)

    def check_itmax(self):
        """Check if the optimization should continue."""
        return self.ctx.iteration <= self.inputs.itmax.value
