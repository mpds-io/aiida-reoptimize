from typing import Any, Callable, Dict, Type

import ase
from aiida.engine import WorkChain

from aiida_reoptimize.base.Evaluation import (
    EvalWorkChainProblem,
    EvalWorkChainStructureProblem,
)

from ..structure.dynamic_structure import StructureCalculator


class OptimizerBuilder:
    def __init__(
        self,
        optimizer_workchain: Type[WorkChain],
        evaluator_workchain: Type[WorkChain],
    ):
        self.optimizer_workchain = optimizer_workchain
        self.evaluator_workchain = evaluator_workchain

    def get_optimizer(self) -> WorkChain:
        """Return a ready-to-run optimizer WorkChain instance."""

        class Optimizer(self.optimizer_workchain):
            evaluator_workchain = self.evaluator_workchain

        # Make it importable
        # so aiida does not complain about the class not being found
        Optimizer.__name__ = self.optimizer_workchain.__name__
        Optimizer.__module__ = self.optimizer_workchain.__module__
        return Optimizer

    @staticmethod
    def _make_problem_evaluator(
        problem_workchain: Type[WorkChain],
        extractor: Callable,
        evaluator: Type[WorkChain],
    ) -> Type[WorkChain]:
        """
        Create a new evaluator class based on the provided problem workchain and extractor.
        """  # noqa: E501
        my_problem_workchain = problem_workchain
        my_extractor = extractor

        class UserEvaluator(evaluator):
            problem_workchain = my_problem_workchain
            extractor = staticmethod(my_extractor)

        # Make it importable
        UserEvaluator.__name__ = evaluator.__name__
        UserEvaluator.__module__ = evaluator.__module__

        return UserEvaluator

    @staticmethod
    def _make_bulk_evaluator(
        problem_builder: Type[WorkChain],
        extractor: Callable,
        evaluator: Type[WorkChain],
    ) -> Type[WorkChain]:
        """
        Create a new evaluator class based on the provided calculator workchain and extractor.
        """  # noqa: E501

        my_problem_builder = problem_builder
        my_extractor = extractor

        class UserEvaluator(evaluator):
            problem_builder = my_problem_builder
            extractor = staticmethod(my_extractor)

        # Make it importable
        UserEvaluator.__name__ = evaluator.__name__
        UserEvaluator.__module__ = evaluator.__module__

        return UserEvaluator

    def _get_structure_problem_builder(
        self,
        bulk: ase.Atoms,
        calculator_workchain: Type[WorkChain],
        structure_keyword: str,
        calculator_parameters: Dict[str, Any] = None,
    ) -> StructureCalculator:
        """
        Create a structure calculator/generator based on the provided bulk and calculator workchain.
        """  # noqa: E501
        return StructureCalculator(
            structure=bulk,
            calculator=calculator_workchain,
            parameters=calculator_parameters,
            structure_keyword=structure_keyword,
        )

    @classmethod
    def from_problem(
        cls,
        optimizer_workchain: Type[WorkChain],
        problem_workchain: Type[WorkChain],
        extractor: Callable,
        evaluator_base: Type[WorkChain] = EvalWorkChainProblem,
    ):
        """
        Build an optimizer for a simple function-like problem.
        """
        # Dynamically create an evaluator class
        evaluator_workchain = cls._make_problem_evaluator(
            problem_workchain, extractor, evaluator=evaluator_base
        )

        return cls(
            optimizer_workchain=optimizer_workchain,
            evaluator_workchain=evaluator_workchain,
        )

    @classmethod
    def from_bulk(
        cls,
        optimizer_workchain: Type[WorkChain],
        calculator_workchain: Type[WorkChain],
        extractor: Callable,
        calculator_parameters: Dict[str, Any],
        bulk: ase.Atoms,
        structure_keyword: str = "structure",
        evaluator_base: Type[WorkChain] = EvalWorkChainStructureProblem,
    ):
        """
        Build an optimizer for a structure/materials problem.
        """

        problem_builder = cls._get_structure_problem_builder(
            bulk=bulk,
            calculator_workchain=calculator_workchain,
            structure_keyword=structure_keyword,
            calculator_parameters=calculator_parameters,
        )

        evaluator_workchain = cls._make_bulk_evaluator(
            problem_builder, extractor, evaluator=evaluator_base
        )

        return cls(
            optimizer_workchain=optimizer_workchain,
            evaluator_workchain=evaluator_workchain,
        )
