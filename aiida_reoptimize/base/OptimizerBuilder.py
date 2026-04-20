from typing import Any, Callable, Dict, Type

import ase
from aiida.engine import WorkChain

from aiida_reoptimize.base.Evaluation import (
    EvalWorkChainProblem,
    EvalWorkChainStructureProblem,
)

from ..structure.dynamic_structure import StructureCalculator
from ..structure.MPDS_structure import get_geometry_MPDS


class OptimizerBuilder:
    """Factory for assembling optimizer WorkChains at runtime.

    Combines an optimizer algorithm, evaluator WorkChain, and result extractor
    into a single ready-to-run WorkChain class.

    The generated classes are patched to appear importable to AiiDA, but they
    can only be used with ``run()`` — not ``submit()`` — because the AiiDA
    daemon cannot import dynamically created classes.
    """

    def __init__(self, optimizer_workchain: Type[WorkChain], evaluator_workchain: Type[WorkChain], extractor: Callable):
        self.optimizer_workchain = optimizer_workchain
        self.evaluator_workchain = evaluator_workchain
        self.extractor = extractor

    def get_optimizer(self) -> WorkChain:
        """Return a ready-to-run optimizer WorkChain class.

        Returns:
            A WorkChain subclass that combines the optimizer algorithm with
            the configured evaluator and extractor.
        """

        class Optimizer(self.optimizer_workchain):
            evaluator_workchain = self.evaluator_workchain
            extractor = self.extractor

        # Make it importable
        # so aiida does not complain about the class not being found
        Optimizer.__name__ = self.optimizer_workchain.__name__
        Optimizer.__module__ = self.optimizer_workchain.__module__
        return Optimizer

    @staticmethod
    def _make_problem_evaluator(
        problem_workchain: Type[WorkChain],
        evaluator: Type[WorkChain],
    ) -> Type[WorkChain]:
        """Create an evaluator class that submits the given problem workchain.

        Args:
            problem_workchain: WorkChain class that accepts ``x`` and returns a value.
            evaluator: Base evaluator class (e.g. ``EvalWorkChainProblem``).

        Returns:
            An evaluator WorkChain subclass with ``problem_workchain`` bound.
        """
        my_problem_workchain = problem_workchain

        class UserEvaluator(evaluator):
            problem_workchain = my_problem_workchain

        # Make it importable
        UserEvaluator.__name__ = evaluator.__name__
        UserEvaluator.__module__ = evaluator.__module__

        return UserEvaluator

    @staticmethod
    def _make_bulk_evaluator(
        problem_builder: Type[WorkChain],
        evaluator: Type[WorkChain],
    ) -> Type[WorkChain]:
        """Create an evaluator class that uses a builder factory for structure problems.

        Args:
            problem_builder: A ``BuilderFactory`` instance (e.g. ``StructureCalculator``).
            evaluator: Base evaluator class (e.g. ``EvalWorkChainStructureProblem``).

        Returns:
            An evaluator WorkChain subclass with ``problem_builder`` bound.
        """

        my_problem_builder = problem_builder

        class UserEvaluator(evaluator):
            problem_builder = my_problem_builder

        # Make it importable
        UserEvaluator.__name__ = evaluator.__name__
        UserEvaluator.__module__ = evaluator.__module__

        return UserEvaluator

    @staticmethod
    def _get_structure_problem_builder(
        bulk: ase.Atoms,
        calculator_workchain: Type[WorkChain],
        structure_keyword: tuple,
        calculator_parameters: Dict[str, Any] = None,  # ty:ignore[invalid-parameter-default]
    ) -> StructureCalculator:
        """Create a ``StructureCalculator`` for the given bulk structure.

        Args:
            bulk: ASE Atoms object representing the reference structure.
            calculator_workchain: Calculator WorkChain class to run on each structure.
            structure_keyword: Path to the structure input in the calculator builder.
            calculator_parameters: Additional parameters for the calculator.

        Returns:
            A ``StructureCalculator`` instance ready to generate builders.
        """
        return StructureCalculator(
            structure=bulk,
            calculator=calculator_workchain,
            calculator_parameters=calculator_parameters,
            structure_keyword=structure_keyword,
        )

    @staticmethod
    def _process_MPDS_query(mpds_query: str) -> ase.Atoms:
        """Parse an MPDS query string and fetch the structure.

        Args:
            mpds_query: Query in ``'Formula/space_group_number'`` format (e.g. ``'WS2/194'``).

        Returns:
            ASE Atoms object for the best-matching structure.

        Raises:
            ValueError: If the query format is invalid.
        """

        phase = mpds_query.split("/")
        if len(phase) != 2:
            raise ValueError(
                "MPDS query should be in the format 'Formula/space_group_number'."  # noqa: E501
            )

        formula, sgs = phase
        sgs = int(sgs)
        return get_geometry_MPDS({"formulae": formula, "sgs": sgs})

    @classmethod
    def from_problem(
        cls,
        optimizer_workchain: Type[WorkChain],
        problem_workchain: Type[WorkChain],
        extractor: Callable,
        evaluator_base: Type[WorkChain] = EvalWorkChainProblem,
    ):
        """Build an optimizer for a simple function-like problem.

        Args:
            optimizer_workchain: Optimizer algorithm WorkChain class.
            problem_workchain: WorkChain that accepts ``x`` and returns a value.
            extractor: Result extractor (e.g. ``BasicExtractor``).
            evaluator_base: Base evaluator class (default: ``EvalWorkChainProblem``).

        Returns:
            An ``OptimizerBuilder`` with the assembled evaluator and extractor.
        """
        # Dynamically create an evaluator class
        evaluator_workchain = cls._make_problem_evaluator(problem_workchain, evaluator=evaluator_base)

        return cls(
            optimizer_workchain=optimizer_workchain, evaluator_workchain=evaluator_workchain, extractor=extractor
        )

    @classmethod
    def from_ase(
        cls,
        optimizer_workchain: Type[WorkChain],
        calculator_workchain: Type[WorkChain],
        extractor: Callable,
        calculator_parameters: Dict[str, Any],
        bulk: ase.Atoms,
        structure_keyword: tuple = ("structure",),
        evaluator_base: Type[WorkChain] = EvalWorkChainStructureProblem,
    ):
        """Build an optimizer for a structure/materials problem.

        Args:
            optimizer_workchain: Optimizer algorithm WorkChain class.
            calculator_workchain: Calculator WorkChain class (e.g. ``FleurScfWorkChain``).
            extractor: Result extractor (e.g. ``BasicExtractor``).
            calculator_parameters: Parameters for the calculator workchain.
            bulk: ASE Atoms object representing the reference structure.
            structure_keyword: Path to the structure input in the builder.
            evaluator_base: Base evaluator class (default: ``EvalWorkChainStructureProblem``).

        Returns:
            An ``OptimizerBuilder`` with the assembled evaluator and extractor.
        """

        problem_builder = cls._get_structure_problem_builder(
            bulk=bulk,
            calculator_workchain=calculator_workchain,
            structure_keyword=structure_keyword,
            calculator_parameters=calculator_parameters,
        )

        evaluator_workchain = cls._make_bulk_evaluator(problem_builder, evaluator=evaluator_base)  # ty:ignore[invalid-argument-type]

        return cls(
            optimizer_workchain=optimizer_workchain, evaluator_workchain=evaluator_workchain, extractor=extractor
        )

    @classmethod
    def from_MPDS(
        cls,
        optimizer_workchain: Type[WorkChain],
        calculator_workchain: Type[WorkChain],
        extractor: Callable,
        calculator_parameters: Dict[str, Any],
        mpds_query: str,
        structure_keyword: tuple = ("structure",),
        evaluator_base: Type[WorkChain] = EvalWorkChainStructureProblem,
    ):
        """Build an optimizer from a structure fetched from the MPDS database.

        Args:
            optimizer_workchain: Optimizer algorithm WorkChain class.
            calculator_workchain: Calculator WorkChain class.
            extractor: Result extractor (e.g. ``BasicExtractor``).
            calculator_parameters: Parameters for the calculator workchain.
            mpds_query: MPDS query string in ``'Formula/space_group_number'`` format.
            structure_keyword: Path to the structure input in the builder.
            evaluator_base: Base evaluator class (default: ``EvalWorkChainStructureProblem``).

        Returns:
            An ``OptimizerBuilder`` with the assembled evaluator and extractor.
        """

        bulk = cls._process_MPDS_query(mpds_query)

        problem_builder = cls._get_structure_problem_builder(
            bulk=bulk,  # Bulk will be fetched from MPDS
            calculator_workchain=calculator_workchain,
            structure_keyword=structure_keyword,
            calculator_parameters=calculator_parameters,
        )

        evaluator_workchain = cls._make_bulk_evaluator(problem_builder, evaluator=evaluator_base)  # ty:ignore[invalid-argument-type]

        return cls(
            optimizer_workchain=optimizer_workchain,
            evaluator_workchain=evaluator_workchain,
            extractor=extractor,
        )
