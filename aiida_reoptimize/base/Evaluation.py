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

from aiida_reoptimize.structure.dynamic_structure import (
    ParameterVectorMismatchError,
    StructureCalculator,
    StructureStandardizationError,
)


class BuilderFactory(Protocol):
    """Protocol for helpers that return ready-to-submit process builders."""

    def get_builder(self, target: Any):
        """Return a calculator or workchain builder for the given target.

        Args:
            target: Parameter vector or target description.

        Returns:
            A process builder ready for submission.
        """


class _EvalBaseWorkChain(WorkChain):
    """Base class for evaluator WorkChains that submit batches of targets and collect results.

    Defines the ``targets`` input, the ``evaluate`` / ``result`` outline, and the
    ``evaluation_results`` output.
    """

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
    """Evaluator that submits a dedicated problem workchain for each target.

    The ``problem_workchain`` class attribute must be set to a WorkChain that
    accepts a single ``x`` input (a scalar or list) and returns the objective
    function value.
    """

    # Expect to receive a workchain that accepts a single ``x`` input and
    # returns the objective function value.
    problem_workchain: Type[WorkChain]

    @classmethod
    def define(cls, spec):
        assert cls.problem_workchain is not None, "problem must be set"  # noqa: E501
        super().define(spec)

    def evaluate(self):
        """Submit the problem workchain once for each target value.

        Returns:
            AiiDA ``ToContext`` mapping that collects submitted process futures.
        """

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
    """Evaluator that uses a ``BuilderFactory`` to submit structure-based targets.

    The ``problem_builder`` class attribute must be set to an object implementing
    the ``BuilderFactory`` protocol (e.g. a ``StructureCalculator`` instance).
    """

    # This workchain is designed for generator-like helpers that convert a
    # target description into a ready-to-submit builder.
    problem_builder: BuilderFactory

    @classmethod
    def define(cls, spec):
        assert cls.problem_builder is not None, "problem must be set"  # noqa: E501
        super().define(spec)

    def evaluate(self):
        """For each target, obtain a builder from the factory and submit it.

        Returns:
            AiiDA ``ToContext`` mapping that collects submitted process futures.
        """
        target_values = {}
        targets = self._targets()
        self.report(f"Evaluating given targets: {targets}")
        for i, x in enumerate(targets):
            builder = self.problem_builder.get_builder(x)
            future = self.submit(builder)
            target_values[f"eval_{i}"] = future
        return ToContext(**target_values)


class _StaticEvalStructureBase(WorkChain):
    """Base class for static structure evaluators registered as AiiDA workchains.

    Accepts a ``structure``, ``targets``, ``calculator_parameters``, and an
    optional ``structure_keyword`` input. Subclasses must implement
    ``generate_structures`` and ``evaluate``.
    """

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

        spec.exit_code(
            410,
            "ERROR_INVALID_PARAMETER_VECTOR",
            message="Target parameter vector is incompatible with the structure Bravais lattice.",
        )
        spec.exit_code(
            411,
            "ERROR_STRUCTURE_STANDARDIZATION_FAILED",
            message="Generated structure could not be standardized with spglib.",
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
        """Load Code nodes from a dictionary of label/PK mappings.

        Args:
            code_dict: Dictionary mapping parameter names to code labels (str) or PKs (int).

        Returns:
            Dictionary mapping parameter names to loaded Code/Node objects.

        Raises:
            ValueError: If a code cannot be loaded.
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
        """Pop ``basis_family`` from calculator_parameters and load or create the basis set.

        Args:
            calculator_parameters: Dictionary of calculator parameters.

        Returns:
            Updated calculator parameters with ``basis_family`` replaced by the loaded object.
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
        """Generate distorted structures and submit the calculator workchain.

        Must be implemented by subclasses that need static structure evaluation.
        """
        raise NotImplementedError("Subclasses must implement generate_structures")

    def evaluate(self):
        """Submit each generated structure to the calculator workchain.

        Must be implemented by subclasses to perform the actual evaluation.
        """
        raise NotImplementedError("Subclasses must implement evaluate")

    def result(self):
        """Store process metadata for the submitted structure evaluations."""

        self.out(
            "evaluation_results",
            List(list=self._collect_evaluation_results()).store(),
        )


class StaticEvalLatticeProblem(_StaticEvalStructureBase):
    """Generate distorted lattice structures and evaluate them with a calculator workchain.

    This workchain accepts a structure as an input argument, enabling the creation
    of static evaluators that can be imported by the AiiDA daemon and used across
    different optimization tasks.
    """

    def generate_structures(self):
        """Generate new structures using ``StructureCalculator`` and store builders in context."""

        self.ctx.builders = []
        targets = self._targets()
        calculator_parameters = self.prepare_calculator_parameters()

        structure_calculator = StructureCalculator(
            structure=self.inputs.structure.get_ase(),
            calculator=self.calculator_workchain,
            calculator_parameters=calculator_parameters,
            structure_keyword=tuple(self.inputs.structure_keyword.get_list()),
        )

        for index, x in enumerate(targets):
            try:
                builder = structure_calculator.get_builder(x)
            except ParameterVectorMismatchError as exc:
                self.report(f"Invalid lattice parameter vector at target {index}: {exc}")
                return self.exit_codes.ERROR_INVALID_PARAMETER_VECTOR
            except StructureStandardizationError as exc:
                self.report(f"Could not standardize generated structure at target {index}: {exc}")
                return self.exit_codes.ERROR_STRUCTURE_STANDARDIZATION_FAILED
            self.ctx.builders.append(builder)

    def evaluate(self):
        """Submit the calculator workchain for each generated structure builder."""
        target_values = {}
        # ! XXX The evaluate method submits all workchains simultaneously, may it lead to resource contention?
        for idx, builder in enumerate(self.ctx.builders):
            future = self.submit(builder)
            target_values[f"eval_{idx}"] = future
        return ToContext(**target_values)
