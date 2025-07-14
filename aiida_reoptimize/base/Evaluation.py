from typing import Type

from aiida.engine import ToContext, WorkChain
from aiida.orm import Dict, Int, List, StructureData, load_code, load_node
from aiida.plugins import DataFactory

from aiida_reoptimize.structure.dynamic_structure import StructureCalculator


class __EvalBaseWorkChain(WorkChain):
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

    def evaluate(self):
        """
        Abstract method for particle evaluations (must be implemented).
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def result(self):
        results = []
        for i in range(len(self.inputs.targets)):
            process = self.ctx[f"eval_{i}"]
            res = {
                "pk": process.pk,
                "status": "ok" if process.is_finished_ok else "failed",
            }
            results.append(res)
        self.out("evaluation_results", List(list=results).store())


class EvalWorkChainProblem(__EvalBaseWorkChain):
    """Base class for evaluating objective functions in optimization workflows."""  # noqa: E501

    # Expect to recive a workchain to be optimized, this workchain recive new
    # parameters and returns the objective function value
    problem_workchain: Type[WorkChain]

    @classmethod
    def define(cls, spec):
        assert cls.problem_workchain is not None, "problem must be set"  # noqa: E501
        super().define(spec)

    def evaluate(self):
        target_values = {}
        # This madness appears to be needed to get the correct type
        # for some reason if you pass List[Int] in aiida input it
        # will be transformed into List[int] and, since your workchain
        # x to be Int and not python int, it will crash
        expected_type = self.problem_workchain.spec().inputs["x"].valid_type
        self.report(f"Evaluating given targets: {self.inputs.targets}")
        for idx, x in enumerate(self.inputs.targets):
            x_wrapped = expected_type(x)
            future = self.submit(self.problem_workchain, x=x_wrapped)
            target_values[f"eval_{idx}"] = future
        return ToContext(**target_values)


class EvalWorkChainStructureProblem(__EvalBaseWorkChain):
    # This workchain designed specifically using it
    # with DynamicStructureWorkChainGenerator
    # (Case when you need to modify complex object and not pass some numbers)
    problem_builder: Type[object]

    @classmethod
    def define(cls, spec):
        assert cls.problem_builder is not None, "problem must be set"  # noqa: E501
        super().define(spec)

    def evaluate(self):
        """
        For each x in targets, use the generator to get a builder and submit it.
        """  # noqa: E501
        target_values = {}
        self.report(f"Evaluating given targets: {self.inputs.targets}")
        for i, x in enumerate(self.inputs.targets):
            builder = self.problem_builder.get_builder(x)
            future = self.submit(builder)
            target_values[f"eval_{i}"] = future
        return ToContext(**target_values)


class _StaticEvalStructureBase(WorkChain):
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
            default=lambda: List([
                "structure",
            ]),
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

    def load_codes(self, code_dict: dict):
        """
        Load the calculator workchain code from the provided dictionary.
        """

        if not code_dict:
            raise ValueError("No codes provided in the code dictionary")

        loaded_codes = {}
        for key, value in code_dict.items():
            try:
                if isinstance(value, str):
                    loaded_codes[key] = load_code(value)
                elif isinstance(value, int):
                    loaded_codes[key] = load_node(value)
                else:
                    raise ValueError(
                        f"Unsupported code format for {key}: {value}"
                    )
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
                basis_family, _ = DataFactory(
                    "crystal_dft.basis_family"
                ).get_or_create(basis_name)
            except Exception as e:
                self.report(f"Error loading basis set {basis_name}: {e}")
                raise e
            calculator_parameters["basis_family"] = basis_family
        return calculator_parameters

    def generate_structures(self):
        """
        Generate structures based on the input structure and targets.
        This method should be implemented in subclasses to modify the structure.
        """
        raise NotImplementedError(
            "Subclasses must implement generate_structures"
        )

    def evaluate(self):
        """
        Evaluate the generated structures using the specified workchain.
        This method should be implemented in subclasses to perform the evaluation.
        """
        raise NotImplementedError("Subclasses must implement evaluate")

    def result(self):
        results = []
        for i in range(len(self.inputs.targets)):
            process = self.ctx[f"eval_{i}"]
            res = {
                "pk": process.pk,
                "status": "ok" if process.is_finished_ok else "failed",
            }
            results.append(res)
        self.out("evaluation_results", List(list=results).store())


class StaticEvalLatticeProblem(_StaticEvalStructureBase):
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
        targets = self.inputs.targets.get_list()

        calculator_parameters = self.inputs.calculator_parameters.get_dict()
        codes = calculator_parameters.pop("codes", {})
        loaded_codes = self.load_codes(codes)
        calculator_parameters.update(loaded_codes)
        calculator_parameters = self.handle_basis_family(calculator_parameters)

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


if __name__ == "__main__":
    import aiida
    from aiida.engine import run
    from aiida.orm import Int, load_node

    aiida.load_profile()

    # Defining the basic extractor
    def result_extractor(results_list, output_key="energy", penalty=1e10):
        """
        Extracts results from a list of dicts with 'pk' and 'status'.
        If status is 'ok', loads the node and returns its output value.
        If status is not 'ok', returns the penalty value.
        """
        extracted = []
        for item in results_list:
            if item["status"] == "ok":
                node = load_node(item["pk"])
                # Assumes the output is a single value node (e.g., Int, Float)
                value = node.outputs[output_key].value
                extracted.append(value)
            else:
                extracted.append(penalty)
        return extracted

    class DummyProblemWorkChain(WorkChain):
        @classmethod
        def define(cls, spec):
            super().define(spec)
            spec.input("x", valid_type=Int)

            spec.outline(cls.evaluate)

            spec.output("energy", valid_type=Int)

        def evaluate(self):
            self.out("energy", Int(self.inputs.x.value**2).store())

    class DummyGenerator:
        def get_builder(self, x):
            builder = DummyProblemWorkChain.get_builder()
            # You have to be sure that you send value of correct type
            builder.x = Int(x)
            return builder

    # Run the EvalWorkChainProblem with the DummyProblemWorkChain
    class UserEvalWorkChainProblem(EvalWorkChainProblem):
        problem_workchain = DummyProblemWorkChain

    int_list = List(list=[i for i in range(5)])  # Example list of x values
    result = run(UserEvalWorkChainProblem, targets=int_list)
    print(result["evaluation_results"])
    print(
        result_extractor(
            result["evaluation_results"], output_key="energy", penalty=1e10
        )
    )

    # Run the EvalWorkChainStructureProblem with the DummyGenerator
    class UserEvalWorkChainStructureProblem(EvalWorkChainStructureProblem):
        problem_builder = DummyGenerator()

    result = run(UserEvalWorkChainStructureProblem, targets=int_list)
    print(result["evaluation_results"])
    print(
        result_extractor(
            result["evaluation_results"], output_key="energy", penalty=1e10
        )
    )
