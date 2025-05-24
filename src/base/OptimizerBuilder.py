from typing import Type

import ase
from aiida.engine import WorkChain
from src.structure.dynamic_structure import DynamicStructure


class OptimizerBuilder:
    def __init__(
        self,
        optimizer_workchain: Type[WorkChain],  # Optimization workchain
        evaluator: Type[WorkChain],  # Initialized evaluator
        optimizer_parameters: dict,  # Optimization parameters
    ):
        self.optimizer_workchain = optimizer_workchain
        self.evaluator_workchain = evaluator
        self.optimizer_parameters = optimizer_parameters

    def get_optimizer(self) -> WorkChain:
        class Optimizer(self.optimizer_workchain):
            evaluator_workchain = self.evaluator_workchain

        return Optimizer(**self.optimizer_parameters)

    def __build_evaluator_problem(
        self,
        evaluator_workchain: Type[WorkChain],
        problem_workchain: Type[WorkChain],
        exctractor: Type[callable],
    ) -> WorkChain:
        class Evaluator(evaluator_workchain):
            problem_builder = problem_workchain
            extractor = staticmethod(exctractor)
        return Evaluator


    def __build_evaluator_structure(
        self,
        evaluator_workchain: Type[WorkChain],
        calculator_workchain: Type[WorkChain],
        extractor: Type[callable],
        structure: Type[ase.Atoms],
        structure_keyword: str,
        calculator_parameters: dict,
    ) -> WorkChain:
        pass

    @classmethod
    def from_problem(
        cls,
        optimizer_workchain: Type[WorkChain],
        calculator_workchain: Type[WorkChain],
        evaluator_workchain: Type[WorkChain],
        exctractor: Type[callable],
        optimizer_parameters: dict
    ):
        pass

    @classmethod
    def from_structure(
        cls,
        optimizer_workchain: Type[WorkChain],
        calculator_workchain: Type[WorkChain],
        evaluator_workchain: Type[WorkChain],
        extractor: Type[callable],
        calculator_parameters: dict,
        optimizer_parameters: dict,
        structure: Type[ase.Atoms],
        structure_keyword: str,
    ):
        pass
