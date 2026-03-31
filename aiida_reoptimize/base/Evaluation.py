"""Reusable evaluation workchains for optimization workflows.

This module contains generic AiiDA workchains that submit a batch of target
evaluations and collect the process identifiers and statuses of the launched
sub-processes. Static structure evaluators additionally generate modified
structures before dispatching the calculator workchains.
"""

from typing import Any, Protocol, Type

from aiida.engine import ToContext, WorkChain
from aiida.orm import Dict, List, StructureData, load_code, load_node
from aiida.plugins import DataFactory

from aiida_reoptimize.structure.dynamic_structure import StructureCalculator


class BuilderFactory(Protocol):
    """Protocol for helpers returning ready-to-submit builders."""

    def get_builder(self, target: Any):
        """Build a calculator or workchain builder for a target."""


class _EvalBaseWorkChain(WorkChain):
    """Common result-collection logic for batched evaluator workchains."""

    @classmethod
    def define(cls, spec):
        """Specify inputs, outputs, and the workchain outline."""
        super().define(spec)
        # It only works if targets is a list of lists.
        # In other cases it crashes.
        spec.input(
            "targets",
            valid_type=List,
            help="List of structural parameter sets to evaluate",
        )

        spec.outline(cls.evaluate, cls.result)

        spec.output(
            "evaluation_results",
            valid_type=List,
            help="List of evaluation results for each target",
        )

    def _targets(self) -> list[Any]:
        """Return targets as a plain Python list."""

        return self.inputs.targets.get_list()

    def _collect_evaluation_results(self) -> list[dict[str, Any]]:
        """Collect process metadata for all submitted evaluations."""

        results = []
        for index, _ in enumerate(self._targets()):
            process = self.ctx[f"eval_{index}"]
            results.append(
                {
                    "pk": process.pk,
                    "status": "ok" if process.is_finished_ok else "failed",
                }
            )
        return results

    def evaluate(self):
        """
        Abstract method for particle evaluations (must be implemented).
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def result(self):
        """Store process metadata for the submitted target evaluations."""

        self.out(
            "evaluation_results",
            List(list=self._collect_evaluation_results()).store(),
        )


class EvalWorkChainProblem(_EvalBaseWorkChain):
    """Evaluate plain parameter targets with a dedicated problem workchain."""

    # Expect to receive a workchain that accepts a single ``x`` input and
    # returns the objective function value.
    problem_workchain: Type[WorkChain]

    @classmethod
    def define(cls, spec):
        assert cls.problem_workchain is not None, "problem must be set"  # noqa: E501
        super().define(spec)

    def evaluate(self):
        """Submit the problem workchain once for each target value."""

        target_values = {}
        # This madness appears to be needed to get the correct type
        # for some reason if you pass List[Int] in aiida input it
        # will be transformed into List[int] and, since your workchain
        # x to be Int and not python int, it will crash
        expected_type = self.problem_workchain.spec().inputs["x"].valid_type
        targets = self._targets()
        self.report(f"Evaluating given targets: {targets}")
        for idx, x in enumerate(targets):
            x_wrapped = expected_type(x)
            future = self.submit(self.problem_workchain, x=x_wrapped)
            target_values[f"eval_{idx}"] = future
        return ToContext(**target_values)


class EvalWorkChainStructureProblem(_EvalBaseWorkChain):
    """Evaluate structure-like targets by obtaining builders from a helper."""

    # This workchain is designed for generator-like helpers that convert a
    # target description into a ready-to-submit builder.
    problem_builder: BuilderFactory

    @classmethod
    def define(cls, spec):
        assert cls.problem_builder is not None, "problem must be set"  # noqa: E501
        super().define(spec)

    def evaluate(self):
        """
        For each x in targets, use the generator to get a builder and submit it.
        """  # noqa: E501
        target_values = {}
        targets = self._targets()
        self.report(f"Evaluating given targets: {targets}")
        for i, x in enumerate(targets):
            builder = self.problem_builder.get_builder(x)
            future = self.submit(builder)
            target_values[f"eval_{i}"] = future
        return ToContext(**target_values)


class _StaticEvalStructureBase(WorkChain):
    """Base class for static structure evaluators registered as AiiDA workchains."""

    calculator_workchain: Type[WorkChain]

    @classmethod
    def define(cls, spec):
        """Specify inputs, outputs, and the workchain outline."""
        super().define(spec)
        # It only works if targets is a list of lists.
        # In other cases it crashes.
        spec.input(
            "structure",
            valid_type=StructureData,
            help="Structure to evaluate with the workchain",
        )

        spec.input(
            "structure_keyword",
            valid_type=List,
            default=lambda: List(
                [
                    "structure",
                ]
            ),
            help="Path to the structure input in the calculator builder.",
        )

        spec.input(
            "calculator_parameters",
            valid_type=Dict,
            help="Parameters for the calculator workchain",
        )

        spec.input(
            "targets",
            valid_type=List,
            help="List of structural parameter sets to evaluate",
        )

        spec.outline(cls.generate_structures, cls.evaluate, cls.result)

        spec.output(
            "evaluation_results",
            valid_type=List,
            help="List of evaluation results for each target",
        )

    def _targets(self) -> list[Any]:
        """Return structure perturbation targets as a Python list."""

        return self.inputs.targets.get_list()

    def _collect_evaluation_results(self) -> list[dict[str, Any]]:
        """Collect process metadata for all submitted calculator jobs."""

        results = []
        for index, _ in enumerate(self._targets()):
            process = self.ctx[f"eval_{index}"]
            results.append(
                {
                    "pk": process.pk,
                    "status": "ok" if process.is_finished_ok else "failed",
                }
            )
        return results

    def load_codes(self, code_dict: dict):
        """
        Load the calculator workchain code from the provided dictionary.
        """

        loaded_codes = {}
        for key, value in code_dict.items():
            try:
                if isinstance(value, str):
                    loaded_codes[key] = load_code(value)
                elif isinstance(value, int):
                    loaded_codes[key] = load_node(value)
                else:
                    raise ValueError(f"Unsupported code format for {key}: {value}")
            except Exception as e:
                raise ValueError(f"Failed to load code for {key}: {e}") from e
        return loaded_codes

    def handle_basis_family(self, calculator_parameters):
        """
        Perform an action only if 'basis_family' is present in calculator_parameters.
        """
        basis_name = calculator_parameters.pop("basis_family", None)
        if basis_name:
            self.report(f"Handling given basis set {basis_name}")
            try:
                basis_family, _ = DataFactory("crystal_dft.basis_family").get_or_create(basis_name)
            except Exception as e:
                self.report(f"Error loading basis set {basis_name}: {e}")
                raise e
            calculator_parameters["basis_family"] = basis_family
        return calculator_parameters

    def prepare_calculator_parameters(self) -> dict[str, Any]:
        """Resolve loadable codes and optional basis families."""

        calculator_parameters = self.inputs.calculator_parameters.get_dict()
        codes = calculator_parameters.pop("codes", {})
        if codes:
            calculator_parameters.update(self.load_codes(codes))
        return self.handle_basis_family(calculator_parameters)

    def generate_structures(self):
        """
        Generate structures based on the input structure and targets.
        This method should be implemented in subclasses to modify the structure.
        """
        raise NotImplementedError("Subclasses must implement generate_structures")

    def evaluate(self):
        """
        Evaluate the generated structures using the specified workchain.
        This method should be implemented in subclasses to perform the evaluation.
        """
        raise NotImplementedError("Subclasses must implement evaluate")

    def result(self):
        """Store process metadata for the submitted structure evaluations."""

        self.out(
            "evaluation_results",
            List(list=self._collect_evaluation_results()).store(),
        )


class StaticEvalLatticeProblem(_StaticEvalStructureBase):
    """Generate distorted structures and evaluate them with a static workchain."""

    def generate_structures(self):
        """
        Generate new structures and builders using StructureCalculator.
        This workflow is needed in order to create static evaluators based on it,
        i.e., such evaluators. Unlike EvalWorkChainStructureProblem,
        which is rigidly tied to a specific structure at creation time,
        this workflow accepts a structure as an argument, which allows
        the creation of static evaluators that can be used in different tasks,
        and can also be imported by the AiiDA daemon.
        """

        self.ctx.builders = []
        targets = self._targets()
        calculator_parameters = self.prepare_calculator_parameters()

        structure_calculator = StructureCalculator(
            structure=self.inputs.structure.get_ase(),
            calculator=self.calculator_workchain,
            calculator_parameters=calculator_parameters,
            structure_keyword=tuple(self.inputs.structure_keyword.get_list()),
        )

        for x in targets:
            builder = structure_calculator.get_builder(x)
            self.ctx.builders.append(builder)

    def evaluate(self):
        """
        Submit the calculator workchain for each generated structure.
        """
        target_values = {}
        # ! XXX The evaluate method submits all workchains simultaneously, may it lead to resource contention?
        for idx, builder in enumerate(self.ctx.builders):
            future = self.submit(builder)
            target_values[f"eval_{idx}"] = future
        return ToContext(**target_values)
