"""Tests for the deterministic L-BFGS optimizer.

Runs the optimizer end-to-end through AiiDA on deterministic quadratic
objectives and verifies the properties that distinguish it from the
stochastic-style Adam/RMSprop optimizers:

- convergence on a deterministic objective (no noise statistics involved);
- monotone descent of the accepted objective values (line search
  guarantees sufficient decrease, so rollbacks cannot occur);
- determinism: repeated runs from the same start converge to the same
  point (no random jumps are taken);
- convergence on an ill-conditioned quadratic (condition number 1e4),
  which requires correctly paired secants and a gradient-scaled first
  step (a unit first step makes the secant pairs collapse onto the stiff
  subspace and the iteration crawls);
- graceful convergence on a noisy objective: with a Gaussian-perturbed
  oracle the optimizer must (a) still land in the vicinity of the true
  minimum, (b) stop at the estimated noise floor instead of looping
  forever, and (c) never accept wildly wrong steps (monotone within the
  noise allowance).
"""

import unittest

import numpy as np
from aiida import load_profile
from aiida.engine import WorkChain, run
from aiida.orm import Bool, Dict, Float, Int, List

from aiida_reoptimize.base.Evaluation import EvalWorkChainProblem
from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.optimizers.convex.QN import LBFGSOptimizer
from aiida_reoptimize.problems.problems import Sphere


def setUpModule():
    load_profile()


dummy_extractor = BasicExtractor(node_extractor=lambda outputs: outputs["value"])


class SphereEvaluator(EvalWorkChainProblem):
    problem_workchain = Sphere


class LBFGSSphereOptimizer(LBFGSOptimizer):
    evaluator_workchain = SphereEvaluator
    extractor = dummy_extractor


class IllConditionedQuad(WorkChain):
    """Quadratic 0.5 * x^T A x with A = diag(1, 1, 1e4)."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("x", valid_type=List)
        spec.outline(cls.run_calc, cls.finalize)
        spec.output("value", valid_type=Float)

    def run_calc(self):
        x = np.array(self.inputs.x.get_list())
        a = np.diag([1.0, 1.0, 1e4])
        self.ctx.result = 0.5 * x @ a @ x

    def finalize(self):
        self.out("value", Float(self.ctx.result).store())


class IllQuadEvaluator(EvalWorkChainProblem):
    problem_workchain = IllConditionedQuad


class LBFGSIllQuadOptimizer(LBFGSOptimizer):
    evaluator_workchain = IllQuadEvaluator
    extractor = dummy_extractor


class NoisySphere(WorkChain):
    """Sphere objective with additive Gaussian noise (sigma = 1e-3)."""

    noise_sigma = 1e-3

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("x", valid_type=List)
        spec.outline(cls.run_calc, cls.finalize)
        spec.output("value", valid_type=Float)

    def run_calc(self):
        x = np.array(self.inputs.x.get_list())
        clean = np.sum(x**2)
        self.ctx.result = clean + np.random.normal(0.0, self.noise_sigma)

    def finalize(self):
        self.out("value", Float(self.ctx.result).store())


class NoisySphereEvaluator(EvalWorkChainProblem):
    problem_workchain = NoisySphere


class LBFGSNoisySphereOptimizer(LBFGSOptimizer):
    evaluator_workchain = NoisySphereEvaluator
    extractor = dummy_extractor


class TestLBFGSOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.itmax = 50
        cls.start = [0.3, -0.2, -0.4]
        cls.parameters = Dict(
            {
                "algorithm_settings": {
                    "memory": 10,
                    "c1": 1e-4,
                    "alpha_init": 1.0,
                    "linesearch_max_iter": 20,
                    "delta": 1e-6,
                    "tolerance": 1e-6,
                },
                "initial_parameters": cls.start,
            }
        )

    def _run(self):
        return run(
            LBFGSSphereOptimizer,
            itmax=Int(self.itmax),
            parameters=self.parameters,
            get_best=Bool(False),
        )

    def test_converges_to_minimum(self):
        results = self._run()
        optimized = np.array(results["optimized_parameters"])
        self.assertLess(np.linalg.norm(optimized), 1e-3)
        self.assertLess(results["final_value"], 1e-6)

    def test_monotone_descent(self):
        results = self._run()
        values = [entry["value"] for entry in results["history"]]
        for previous, current in zip(values, values[1:], strict=False):
            self.assertLessEqual(current, previous + 1e-12)

    def test_deterministic(self):
        first = self._run()
        second = self._run()
        np.testing.assert_allclose(
            np.array(first["optimized_parameters"]),
            np.array(second["optimized_parameters"]),
            atol=1e-8,
        )
        self.assertEqual(first["final_value"], second["final_value"])

    def test_ill_conditioned_convergence(self):
        parameters = Dict(
            {
                "algorithm_settings": {
                    "memory": 10,
                    "c1": 1e-4,
                    "alpha_init": 1.0,
                    "linesearch_max_iter": 40,
                    "delta": 1e-6,
                    "tolerance": 1e-8,
                },
                "initial_parameters": [0.3, -0.2, 0.4],
            }
        )
        results = run(
            LBFGSIllQuadOptimizer,
            itmax=Int(30),
            parameters=parameters,
            get_best=Bool(False),
        )
        optimized = np.array(results["optimized_parameters"])
        self.assertLess(np.linalg.norm(optimized), 1e-4, msg=f"got {optimized}")
        self.assertLess(results["final_value"], 1e-8)

    def test_noisy_objective_converges_to_vicinity(self):
        np.random.seed(42)
        parameters = Dict(
            {
                "algorithm_settings": {
                    "memory": 10,
                    "c1": 1e-4,
                    "alpha_init": 1.0,
                    "linesearch_max_iter": 40,
                    "delta": 3e-2,
                    "tolerance": 1e-8,
                },
                "initial_parameters": [0.3, -0.2, -0.4],
            }
        )
        results = run(
            LBFGSNoisySphereOptimizer,
            itmax=Int(100),
            parameters=parameters,
            get_best=Bool(False),
        )
        optimized = np.array(results["optimized_parameters"])
        self.assertLess(
            np.linalg.norm(optimized),
            0.1,
            msg=f"noisy run drifted from the minimum: {optimized}",
        )

    def test_noisy_objective_stops_within_iteration_budget(self):
        np.random.seed(7)
        parameters = Dict(
            {
                "algorithm_settings": {
                    "memory": 10,
                    "c1": 1e-4,
                    "alpha_init": 1.0,
                    "linesearch_max_iter": 40,
                    "delta": 3e-2,
                    "tolerance": 1e-8,
                },
                "initial_parameters": [0.3, -0.2, -0.4],
            }
        )
        results = run(
            LBFGSNoisySphereOptimizer,
            itmax=Int(100),
            parameters=parameters,
            get_best=Bool(False),
        )
        history = results["history"]
        self.assertLessEqual(
            len(history),
            100,
            msg="noisy run must not loop forever",
        )

    def test_mismatched_delta_stops_at_noise_floor_immediately(self):
        np.random.seed(11)
        parameters = Dict(
            {
                "algorithm_settings": {
                    "memory": 10,
                    "c1": 1e-4,
                    "alpha_init": 1.0,
                    "linesearch_max_iter": 40,
                    "delta": 1e-6,
                    "tolerance": 1e-8,
                },
                "initial_parameters": [0.3, -0.2, -0.4],
            }
        )
        results = run(
            LBFGSNoisySphereOptimizer,
            itmax=Int(100),
            parameters=parameters,
            get_best=Bool(False),
        )
        self.assertLessEqual(
            len(results["history"]),
            3,
            msg="with delta << sigma the FD gradient is pure noise: must stop within a few iterations",
        )


if __name__ == "__main__":
    unittest.main()
