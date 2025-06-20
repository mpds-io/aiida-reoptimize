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
    def __init__(
        self,
        optimizer_workchain: Type[WorkChain],
        evaluator_workchain: Type[WorkChain],
        extractor: Type[Callable]
    ):
        self.optimizer_workchain = optimizer_workchain
        self.evaluator_workchain = evaluator_workchain
        self.extractor = extractor

    def get_optimizer(self) -> WorkChain:
        """Return a ready-to-run optimizer WorkChain instance."""

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
        """
        Create a new evaluator class based on the provided problem workchain and extractor.
        """  # noqa: E501
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
        """
        Create a new evaluator class based on the provided calculator workchain and extractor.
        """  # noqa: E501

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

    @staticmethod
    def _process_MPDS_query(mpds_query: str) -> ase.Atoms:
        """
        Process the MPDS query to get the structure.
        This function should be implemented to fetch the structure from MPDS.
        It excepts query like 'Formula/space_group_number'
        for example 'WS2/194'.
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
        extractor: Type[Callable],
        evaluator_base: Type[WorkChain] = EvalWorkChainProblem,
    ):
        """
        Build an optimizer for a simple function-like problem.
        """
        # Dynamically create an evaluator class
        evaluator_workchain = cls._make_problem_evaluator(
            problem_workchain, evaluator=evaluator_base
        )

        return cls(
            optimizer_workchain=optimizer_workchain,
            evaluator_workchain=evaluator_workchain,
            extractor=extractor
        )

    @classmethod
    def from_ase(
        cls,
        optimizer_workchain: Type[WorkChain],
        calculator_workchain: Type[WorkChain],
        extractor: Type[Callable],
        calculator_parameters: Dict[str, Any],
        bulk: ase.Atoms,
        structure_keyword: tuple = ("structure",),
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
            problem_builder, evaluator=evaluator_base
        )

        return cls(
            optimizer_workchain=optimizer_workchain,
            evaluator_workchain=evaluator_workchain,
            extractor=extractor
        )

    @classmethod
    def from_MPDS(
        cls,
        optimizer_workchain: Type[WorkChain],
        calculator_workchain: Type[WorkChain],
        extractor: Type[Callable],
        calculator_parameters: Dict[str, Any],
        mpds_query: str,
        structure_keyword: tuple = ("structure",),
        evaluator_base: Type[WorkChain] = EvalWorkChainStructureProblem,
    ):
        """
        Build an optimizer for a structure/materials problem based on MPDS.
        """

        bulk = cls._process_MPDS_query(mpds_query)

        problem_builder = cls._get_structure_problem_builder(
            bulk=bulk,  # Bulk will be fetched from MPDS
            calculator_workchain=calculator_workchain,
            structure_keyword=structure_keyword,
            calculator_parameters=calculator_parameters,
        )

        evaluator_workchain = cls._make_bulk_evaluator(
            problem_builder, evaluator=evaluator_base
        )

        return cls(
            optimizer_workchain=optimizer_workchain,
            evaluator_workchain=evaluator_workchain,
            extractor=extractor,
        )
