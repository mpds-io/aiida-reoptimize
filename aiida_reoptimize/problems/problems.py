import numpy as np
from aiida.engine import WorkChain
from aiida.orm import Float, List

__all__ = ["Ackley", "Rastring", "Sphere"]


class _basicProblem(WorkChain):
    """Base WorkChain for benchmark optimization problems.

    Defines a single ``x`` input (List) and a ``value`` output (Float).
    Subclasses must implement ``run_calc`` to compute the objective value.
    """

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
    """Ackley function benchmark problem.

    A widely-used multimodal test function with many local minima.
    Global minimum at ``x = 0`` with value ``0``.

    See: https://en.wikipedia.org/wiki/Ackley_function
    """

    def run_calc(self):
        x = np.array(self.inputs.x.get_list())
        n = x.size
        # default values from https://en.wikipedia.org/wiki/Ackley_function
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        self.ctx.result = -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e


class Rastring(_basicProblem):
    """Rastrigin function benchmark problem.

    A non-convex function with many local minima used as a performance test
    for optimization algorithms. Global minimum at ``x = 0`` with value ``0``.

    See: https://en.wikipedia.org/wiki/Rastrigin_function
    """

    def run_calc(self):
        x = np.array(self.inputs.x.get_list())
        n = x.size
        A = 10
        self.ctx.result = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


class Sphere(_basicProblem):
    """Sphere function benchmark problem.

    The simplest convex test function. Global minimum at ``x = 0`` with value ``0``.
    """

    def run_calc(self):
        x = np.array(self.inputs.x.get_list())
        self.ctx.result = np.sum(x**2)
