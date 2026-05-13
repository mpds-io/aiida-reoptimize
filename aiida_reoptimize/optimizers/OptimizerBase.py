from typing import Callable, Type

from aiida.engine import WorkChain, run_get_node
from aiida.orm import Bool, Dict, Float, Int, List, StructureData


class _OptimizerBase(WorkChain):
    """Base class for all optimization algorithm WorkChains.

    Defines common inputs (``parameters``, ``itmax``, ``get_best``, ``structure``),
    common outputs (``optimized_parameters``, ``final_value``, ``history``,
    ``result_node_pk``), and the ``initialize`` / ``optimization_process`` /
    ``finalize`` outline.

    Subclasses must set the ``evaluator_workchain`` and ``extractor`` class
    attributes and implement ``optimization_process`` and ``finalize``.
    """

    evaluator_workchain: Type[WorkChain]
    extractor: Callable

    @classmethod
    def define(cls, spec):
        assert cls.evaluator_workchain is not None, "evaluator must be set"  # noqa: E501
        assert cls.extractor is not None, "extractor must be set"

        super().define(spec)
        spec.input("parameters", valid_type=Dict, help="Optimization parameters.")
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

        spec.exit_code(
            430,
            "ERROR_EVALUATOR_FAILED",
            message="Evaluator WorkChain failed before returning evaluation results.",
        )
        spec.exit_code(
            431,
            "ERROR_INVALID_OPTIMIZATION_PARAMETERS",
            message="Optimization parameters are incompatible with the optimizer or input structure.",
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

    def initialize(self):
        self.ctx.history = []

    def record_history(self, iteration, value, result_node_pk=None):
        entry = {
            "iteration": iteration,
            "value": value,
            "result_node_pk": result_node_pk,
        }
        self.ctx.history.append(entry)
        return entry

    def report_progress(self):
        if not self.ctx.history:
            return
        entry = self.ctx.history[-1]
        parts = [f"Iteration {entry['iteration']}", f"value={entry['value']:.6e}"]
        if entry.get("result_node_pk") is not None:
            parts.append(f"pk={entry['result_node_pk']}")
        self.report(" | ".join(parts))

    def optimization_process(self):
        """Main optimization loop."""
        raise NotImplementedError("Subclasses must implement optimization_process()")

    def finalize(self):
        """Finalize the optimization process. Subclasses must override this method."""
        raise NotImplementedError("Subclasses must implement finalize()")

    def run_evaluator(self, targets, **kwargs):
        """Run the evaluator workchain with the given targets.

        Args:
            targets: AiiDA ``List`` of parameter vectors to evaluate.
            **kwargs: Additional keyword arguments passed to the evaluator
                (e.g. ``calculator_parameters``).

        Returns:
            Dictionary of evaluator outputs including ``evaluation_results``, or
            ``None`` when the evaluator failed.
        """
        if self.inputs.get("structure"):
            outputs, node = run_get_node(
                self.evaluator_workchain,
                targets=targets,
                structure=self.inputs.structure,
                **kwargs,
            )
        else:
            outputs, node = run_get_node(self.evaluator_workchain, targets=targets, **kwargs)

        self.ctx.last_evaluator_node_pk = node.pk

        if not node.is_finished_ok:
            exit_status = node.exit_status
            exit_message = node.exit_message or "No exit message was provided."
            self.report(
                f"Evaluator {self.evaluator_workchain.__name__} failed "
                f"with exit status {exit_status} (pk={node.pk}): {exit_message}"
            )
            return None

        if "evaluation_results" not in outputs:
            self.report(
                f"Evaluator {self.evaluator_workchain.__name__} finished without "
                f"the required 'evaluation_results' output (pk={node.pk})."
            )
            return None

        return outputs

    def check_itmax(self):
        """Check if the current iteration is within the maximum limit.

        Returns:
            True if the optimizer should continue iterating.
        """
        return self.ctx.iteration <= self.inputs.itmax.value
