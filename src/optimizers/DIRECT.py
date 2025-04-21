import numpy as np
from aiida.engine import WorkChain, while_
from aiida.orm import ArrayData, Float, Int, List, Str


class DIRECTWorkChain(WorkChain):
    """
    This WorkChain is designed to perform optimization using the DIRECT (DIviding RECTangles) algorithm.
    It divides the search space into hyperrectangles, evaluates the objective function, and iteratively
    refines the search space to find the optimal solution.

    Attributes:
        dimensions (Int): Dimensionality of the parameter space.
        bounds (List): Search space boundaries [min, max] for each dimension.
        key_value (Str): Identifier for the result key in sub-process outputs.
        max_iterations (Int): Maximum number of optimization cycles.
        epsilon (Float): Convergence criteria for the optimization.
        penalty (Float): Penalty value for failed calculations.

    Outputs:
        final_value (Float): The best value of the objective function found during optimization.
        optimized_parameters (List): The parameters corresponding to the best value.
    """  # noqa: E501

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "dimensions",
            valid_type=Int,
            default=Int(1),
            help="Dimensionality of parameter space",
        )
        spec.input(
            "bounds",
            valid_type=List,
            default=List(list=[[3.7, 4.2]]),
            help="Search space boundaries [min, max] per dimension",
        )
        spec.input(
            "key_value",
            valid_type=Str,
            default=lambda: Str("energy"),
            help="Result key identifier for sub-process outputs",
        )
        spec.input(
            "max_iterations",
            valid_type=Int,
            default=lambda: Int(100),
            help="Maximum number of optimization cycles",
        )
        spec.input(
            "epsilon",
            valid_type=Float,
            default=lambda: Float(1e-1),
            help="Convergence criteria",
        )
        spec.input(
            "penalty",
            valid_type=Float,
            default=lambda: Float(1e8),
            help="Penalty value for failed calculations",
        )

        spec.outline(
            cls.initialize,
            while_(cls.continue_condition)(
                cls.select_potentially_optimal,
                cls.divide_rectangles,
                cls.evaluate,
                cls.collect_results,
                cls.update_best,
            ),
            cls.finalize,
        )

        spec.output("final_value", valid_type=Float)
        spec.output("optimized_parameters", valid_type=List)

    def initialize(self):
        """Инициализация начальных параметров"""
        self.ctx.dimensions = self.inputs.dimensions.value
        self.ctx.bounds = np.array(self.inputs.bounds.get_list())

        self.ctx.iteration = 0
        self.ctx.best_value = self.inputs.penalty.value + 1
        self.ctx.best_solution = np.zeros(self.ctx.dim)
        self.ctx.should_continue = True

        # Inital heperrectangle
        initial_rect = ArrayData()
        # XXX Add bounds size check
        lower_bounds = self.ctx.bounds[:, 0]
        upper_bounds = self.ctx.bounds[:, 1]

        initial_rect.set_array("lower", lower_bounds)
        initial_rect.set_array("upper", upper_bounds)
        initial_rect.set_array("center", (lower_bounds + upper_bounds) / 2)
        initial_rect.set_array(
            self.inputs.key_value.value,
            np.array([self.inputs.penalty.value] * self.ctx.dim),
        )
        self.ctx.rectangles = [initial_rect]

    def continue_condition(self):
        """Continue iteration if needed and if the rectangles are not too small"""  # noqa: E501
        return (
            self.ctx.iteration < self.inputs.max_iterations
            and np.max([
                np.max(r.get_array("upper") - r.get_array("lower"))
                for r in self.ctx.rectangles
            ])
            > self.inputs.epsilon
        )

    def select_potentially_optimal(self):
        key = self.inputs.key_value.value
        sorted_rects = sorted(
            self.ctx.rectangles, key=lambda x: x.get_array(key)
        )
        po_rects = []
        min_f = sorted_rects[0].get_array(key)[0]

        # Select all rectangles with f <= min_f + ε*|min_f|
        candidates = [
            r
            for r in sorted_rects
            if r.get_array(key)[0]
            <= min_f + self.inputs.epsilon.value * abs(min_f)
        ]

        # Select largest rectangles among candidates
        max_size = max([
            np.max(r.get_array("upper") - r.get_array("lower"))
            for r in candidates
        ])
        po_rects = [
            r
            for r in candidates
            if np.max(r.get_array("upper") - r.get_array("lower"))
            >= max_size - 1e-12
        ]

        self.ctx.current_rectangles = po_rects

    def divide_rectangles(self):
        new_rects = []
        for rect in self.ctx.current_rectangles:
            lower = rect.get_array("lower")
            upper = rect.get_array("upper")
            dim = np.argmax(upper - lower)
            delta = (upper[dim] - lower[dim]) / 3
            c = lower[dim] + delta  # First third

            # Create 3 new hyperrectangles
            for part in range(3):
                new_lower = lower.copy()
                new_upper = upper.copy()

                if part == 0:
                    new_upper[dim] = c  # Left third
                elif part == 1:
                    new_lower[dim] = c
                    new_upper[dim] = c + delta  # Middle third
                else:
                    new_lower[dim] = c + delta
                    new_upper[dim] = upper[dim]  # Right third

                new_rect = ArrayData()
                new_rect.set_array("lower", new_lower)
                new_rect.set_array("upper", new_upper)
                new_rect.set_array("center", (new_lower + new_upper) / 2)
                new_rect.set_array(
                    self.inputs.key_value.value,
                    np.array([self.inputs.penalty.value]),
                )
                new_rects.append(new_rect)

        self.ctx.new_rectangles = new_rects
        self.ctx.targets = [r.get_array("center") for r in new_rects]
        self.report(
            f"Iteration {self.ctx.iteration}: new rectangles = {self.ctx.targets}"  # noqa: E501
        )

    def evaluate(self):
        """
        Abstract method for particle evaluations (must be implemented).

        Should create sub-processes for each particle and store them in:
        self.ctx[f'eval_{i}'] for i in range(num_particles)
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def collect_results(self):
        key = self.inputs.key_value.value
        updated_rects = []
        for i, rect in enumerate(self.ctx.new_rectangles):
            process = self.ctx[f"eval_{i}"]
            res = (
                process.outputs[key].value
                if process.is_finished_ok
                else self.inputs.penalty.value
            )

            # Create a new ArrayData node
            new_rect = ArrayData()
            new_rect.set_array("lower", rect.get_array("lower"))
            new_rect.set_array("upper", rect.get_array("upper"))
            new_rect.set_array("center", rect.get_array("center"))
            new_rect.set_array(key, np.array([res]))  # Modify before storing

            updated_rects.append(new_rect)

        # Replace old rectangles with new ones in the context
        self.ctx.new_rectangles = updated_rects

    def update_best(self):
        """Update the best solution found so far"""
        for rect in self.ctx.new_rectangles:
            value = rect.get_array(self.inputs.key_value.value)[0]
            if value < self.ctx.best_value:
                self.previous_best_value = self.ctx.best_value
                self.ctx.best_value = value
                self.ctx.best_solution = rect.get_array("center")

        # Update rectangles (Remove current rectangles from rectangles)
        self.ctx.rectangles = [
            r
            for r in self.ctx.rectangles
            if r not in self.ctx.current_rectangles
        ] + self.ctx.new_rectangles
        self.ctx.iteration += 1

        self.report(
            f"Iteration {self.ctx.iteration}: best value = {self.ctx.best_value}, best solution = {self.ctx.best_solution}"  # noqa: E501
        )

    def finalize(self):
        """Финальные действия"""
        self.out("final_value", Float(self.ctx.best_value).store())
        self.out(
            "optimized_parameters",
            List(self.ctx.best_solution.tolist()).store(),
        )
