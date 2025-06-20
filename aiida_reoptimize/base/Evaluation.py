from typing import Type

from aiida.engine import ToContext, WorkChain
from aiida.orm import Int, List


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
