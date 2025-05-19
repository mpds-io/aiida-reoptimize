from typing import Type

from aiida.engine import ToContext, WorkChain
from aiida.orm import List, Str


class EvalWorkChainProblem(WorkChain):
    """Base class for evaluating objective functions in optimization workflows."""  # noqa: E501

    problem_workchain: Type[WorkChain]

    # Expect to recive a workchain to be optimized, this workchain recive new
    # parameters and returns the objective function value

    @classmethod
    def define(cls, spec):
        """Specify inputs, outputs, and the workchain outline."""
        assert cls.problem_workchain is not None, "problem must be set"  # noqa: E501
        super().define(spec)
        spec.input(
            "targets",
            valid_type=List,
            help="List of structural parameter sets to evaluate",
        )
        spec.input(
            "output_keyword",
            valid_type=Str,
            default=lambda: Str("energy"),
            help="Key for the target value",
        )

        spec.outline(cls.evaluate, cls.result)

        spec.output(
            "evaluation_results",
            valid_type=List,
            help="List of evaluation results for each target",
        )

    def evaluate(self):
        target_values = {}
        for i, pos in enumerate(self.inputs.targets):
            future = self.submit(self.problem_workchain, x=Int(pos))
            target_values[f"eval_{i}"] = future
        return ToContext(
            **target_values
        )  # Wait for all pro   cesses to complete

    def result(self):
        results = []
        for i in range(len(self.inputs.targets)):
            process = self.ctx[f"eval_{i}"]
            results.append(
                process.outputs[self.inputs.output_keyword.value].value
            )
        self.out("evaluation_results", List(list=results).store())


class EvalWorkChainStructureProblem(WorkChain):

    problem_builder: Type[WorkChain]
    generator: Type[object]  # Placeholder for the generator type

    # This workchain designed specifically using it 
    # with DynamicStructureWorkChainGenerator 
    # (Case when you need to modify complex object and not pass some numbers)

    @classmethod
    def define(cls, spec):
        assert cls.problem_builder is not None, "problem_builder must be set"  # noqa: E501
        super().define(spec)
        spec.input(
            "targets",
            valid_type=List,
            help="List of structural parameter sets",
        )
        spec.input(
            "output_keyword",
            valid_type=Str,
            default=lambda: Str("energy"),
            help="Key for the target value",
        )
        spec.outline(
            cls.evaluate,
            cls.result,
        )
        spec.output("evaluation_results", valid_type=List)

    def evaluate(self):
        """
        For each x in targets, use the generator to get a builder and submit it.
        """  # noqa: E501
        target_values = {}
        targets = self.inputs.targets.get_list()

        for i, x in enumerate(targets):
            builder = self.generator.get_builder(Int(x))
            future = self.submit(builder)
            target_values[f"eval_{i}"] = future
        return ToContext(**target_values)

    def result(self):
        results = []
        for i in range(len(self.inputs.targets)):
            process = self.ctx[f"eval_{i}"]
            results.append(
                process.outputs[self.inputs.output_keyword.value].value
            )
        self.out("evaluation_results", List(list=results).store())


if __name__ == "__main__":
    import aiida
    from aiida.engine import run
    from aiida.orm import Int

    aiida.load_profile()

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
            builder.x = Int(x)
            return builder

    # Run the EvalWorkChainProblem with the DummyProblemWorkChain
    class UserEvalWorkChainProblem(EvalWorkChainProblem):
        problem_workchain = DummyProblemWorkChain

    int_list = List([0, 1, 2, 3, 4])
    result = run(UserEvalWorkChainProblem, targets=int_list)
    print(result["evaluation_results"])
    # Output: [0, 1, 4, 9, 16]

    # Run the EvalWorkChainStructureProblem with the DummyGenerator
    class UserEvalWorkChainStructureProblem(EvalWorkChainStructureProblem):
        problem_builder = DummyProblemWorkChain
        generator = DummyGenerator()

    int_list = List([9, 2, 3, 4, 7])
    result = run(UserEvalWorkChainStructureProblem, targets=int_list)
    print(result["evaluation_results"])
    # Output: [81, 4, 9, 16, 49]
