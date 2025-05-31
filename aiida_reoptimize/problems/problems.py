import numpy as np
from aiida.engine import WorkChain
from aiida.orm import Float, List

__all__ = ["Ackley", "Rastring", "Sphere"]


class _basicProblem(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("x", valid_type=List)
        spec.outline(cls.run_calc, cls.finalize)
        spec.output("value", valid_type=Float)

    def run_calc(self):
        """Calculate the value of the objective function."""
        raise NotImplementedError("Subclasses must implement value_calc()")

    def finalize(self):
        self.out("value", Float(self.ctx.result).store())


class Ackley(_basicProblem):
    """
    Implementation of Ackley function for optimization problems. Global minimum at (0) with value 0.
    ref https://en.wikipedia.org/wiki/Ackley_function
    """  # noqa: E501

    def run_calc(self):
        x = np.array(self.inputs.x.get_list())
        n = x.size
        # default values from https://en.wikipedia.org/wiki/Ackley_function
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        self.ctx.result = (
            -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e
        )


class Rastring(_basicProblem):
    """
    Implementation of Rastrigin function for optimization problems. Global minimum at (0) with value 0.
    ref https://en.wikipedia.org/wiki/Rastrigin_function
    """  # noqa: E501

    def run_calc(self):
        x = np.array(self.inputs.x.get_list())
        n = x.size
        A = 10
        self.ctx.result = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


class Sphere(_basicProblem):
    """Implementation of Sphere function for optimization problems. Global minimum at (0) with value 0."""  # noqa: E501

    def run_calc(self):
        x = np.array(self.inputs.x.get_list())
        self.ctx.result = np.sum(x**2)
