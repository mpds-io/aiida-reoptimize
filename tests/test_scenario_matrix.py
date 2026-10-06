"""Scenario tests for the LBFGS optimizer driving the REAL WorkChain.

Every scenario below is an end-to-end AiiDA run of ``LBFGSOptimizer``
against a purpose-built oracle WorkChain, so the checks exercise the
actual implementation: two-loop direction, first-step alpha scaling,
Armijo line search with noise allowance, secant-pair bookkeeping with
relative curvature threshold, noise floor stop, stall/patience stop,
parameter clamping, and the failure paths (crashing oracle, NaN values,
penalty values, flat objective).

Each oracle evaluation in these tests costs milliseconds, so the full
matrix runs in seconds despite going through AiiDA.
"""

import unittest

import numpy as np
from aiida import load_profile
from aiida.engine import WorkChain, run_get_node
from aiida.orm import Bool, Dict, Float, Int, List

from aiida_reoptimize.base.Evaluation import EvalWorkChainProblem
from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.optimizers.convex.QN import LBFGSOptimizer


def setUpModule():
    load_profile()


extractor = BasicExtractor(node_extractor=lambda outputs: outputs["value"])


def make_oracle(objective):
    """Wrap a scalar objective into an AiiDA oracle evaluator pair."""

    class _Problem(WorkChain):
        @classmethod
        def define(cls, spec):
            super().define(spec)
            spec.input("x", valid_type=List)
            spec.outline(cls.run_calc, cls.finalize)
            spec.output("value", valid_type=Float)

        def run_calc(self):
            self.ctx.result = float(objective(np.array(self.inputs.x.get_list())))

        def finalize(self):
            self.out("value", Float(self.ctx.result).store())

    class _Evaluator(EvalWorkChainProblem):
        problem_workchain = _Problem

    return _Evaluator


def make_optimizer(evaluator):
    """Bind an evaluator to a concrete LBFGS optimizer class."""

    class _Optimizer(LBFGSOptimizer):
        evaluator_workchain = evaluator
        extractor = extractor

    return _Optimizer


def run_optimizer(evaluator, initial_parameters, algorithm_settings, itmax=200):
    """Run one scenario end-to-end; return (outputs, process_node)."""

    parameters = Dict(
        {
            "algorithm_settings": algorithm_settings,
            "initial_parameters": initial_parameters,
        }
    )
    return run_get_node(
        make_optimizer(evaluator),
        itmax=Int(itmax),
        parameters=parameters,
        get_best=Bool(False),
    )


DEFAULTS = {
    "memory": 10,
    "c1": 1e-4,
    "alpha_init": 1.0,
    "linesearch_max_iter": 40,
    "delta": 1e-6,
    "tolerance": 1e-6,
}


def sphere(x):
    return float(np.sum(x**2))


def ill_conditioned(x):
    a = np.diag([1.0, 1.0, 1e4])
    return float(0.5 * x @ a @ x)


class TestScenarioConvergence(unittest.TestCase):
    def test_sphere_converges_monotone(self):
        results, node = run_optimizer(
            make_oracle(sphere),
            [0.3, -0.2, -0.4],
            dict(DEFAULTS),
        )
        self.assertEqual(node.exit_status, 0)
        values = [entry["value"] for entry in results["history"]]
        for previous, current in zip(values, values[1:], strict=False):
            self.assertLessEqual(current, previous + 1e-12)
        self.assertLess(np.linalg.norm(results["optimized_parameters"]), 1e-3)

    def test_ill_conditioned_converges(self):
        results, node = run_optimizer(
            make_oracle(ill_conditioned),
            [0.3, -0.2, 0.4],
            dict(DEFAULTS, tolerance=1e-8, itmax=30) | {"itmax": 30},
        )
        self.assertEqual(node.exit_status, 0)
        self.assertLess(np.linalg.norm(results["optimized_parameters"]), 1e-4)

    def test_rosenbrock_converges(self):
        def rosen(x):
            return float((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)

        results, node = run_optimizer(
            make_oracle(rosen),
            [-1.2, 1.0],
            dict(DEFAULTS, tolerance=1e-3, max_step=0.0, itmax=500) | {"itmax": 500},
        )
        self.assertEqual(node.exit_status, 0)
        self.assertLess(
            np.linalg.norm(np.array(results["optimized_parameters"]) - [1.0, 1.0]),
            5e-2,
        )

    def test_one_dimensional_converges(self):
        results, node = run_optimizer(
            make_oracle(lambda x: float((x[0] - 1.5) ** 2)),
            [3.0],
            dict(DEFAULTS),
        )
        self.assertEqual(node.exit_status, 0)
        self.assertLess(abs(results["optimized_parameters"][0] - 1.5), 1e-3)

    def test_at_minimum_immediate_convergence(self):
        results, node = run_optimizer(
            make_oracle(sphere),
            [0.0, 0.0],
            dict(DEFAULTS),
        )
        self.assertEqual(node.exit_status, 0)
        self.assertEqual(len(results["history"]), 0)

    def test_anisotropic_soft_mode_at_tolerance_floor(self):
        """The soft component (curvature 1e-6) has gradient 1e-6*|x1|:
        with tolerance 1e-6 the optimizer must correctly stop with the
        soft mode at its tolerance-implied floor, and a stricter tolerance
        must resolve it."""

        def objective(x):
            return float(0.5 * (1e-6 * x[0] ** 2 + x[1] ** 2))

        results, node = run_optimizer(
            make_oracle(objective),
            [1.0, 1.0],
            dict(DEFAULTS, max_step=0.5, itmax=300) | {"itmax": 300},
        )
        self.assertEqual(node.exit_status, 0)
        self.assertLess(abs(results["optimized_parameters"][1]), 1e-3)

        strict, node2 = run_optimizer(
            make_oracle(objective),
            [1.0, 1.0],
            dict(DEFAULTS, tolerance=1e-9, max_step=0.5, itmax=300) | {"itmax": 300},
        )
        self.assertEqual(node2.exit_status, 0)
        self.assertLess(abs(strict["optimized_parameters"][0]), 1e-3)


class TestScenarioNoise(unittest.TestCase):
    def test_noisy_matched_delta_reaches_vicinity(self):
        np.random.seed(42)

        def noisy(x):
            return float(np.sum(x**2) + np.random.normal(0, 1e-3))

        results, node = run_optimizer(
            make_oracle(noisy),
            [0.3, -0.2, -0.4],
            dict(DEFAULTS, delta=3e-2, tolerance=1e-8, patience=5) | {"itmax": 100},
        )
        self.assertEqual(node.exit_status, 0)
        self.assertLess(np.linalg.norm(results["optimized_parameters"]), 0.1)
        self.assertLessEqual(len(results["history"]), 100)

    def test_noisy_mismatched_delta_stops_at_floor(self):
        np.random.seed(99)

        def noisy(x):
            return float(np.sum(x**2) + np.random.normal(0, 1e-3))

        x0 = [0.3, -0.2, -0.4]
        results, node = run_optimizer(
            make_oracle(noisy),
            x0,
            dict(DEFAULTS, delta=1e-6, tolerance=1e-8, patience=5) | {"itmax": 100},
        )
        self.assertEqual(node.exit_status, 0)
        self.assertLessEqual(len(results["history"]), 3)
        self.assertLess(np.linalg.norm(results["optimized_parameters"] - np.array(x0)), 5e-3)

    def test_penalized_noise_sample_does_not_corrupt_floor(self):
        """One penalty (failed evaluation) inside a duplicated noise probe
        must be filtered out instead of inflating sigma_f to ~1e10 and
        triggering a spurious noise-floor stop."""

        rng = np.random.default_rng(5)
        calls = {"n": 0}

        def occasionally_penalized(x):
            calls["n"] += 1
            # every 7th evaluation "fails" -> extractor penalty
            if calls["n"] % 7 == 0:
                return extractor.get_penalty()
            return float(np.sum(x**2) + rng.normal(0, 1e-3))

        results, node = run_optimizer(
            make_oracle(occasionally_penalized),
            [0.3, -0.2, -0.4],
            dict(DEFAULTS, delta=3e-2, tolerance=1e-8, patience=5) | {"itmax": 100},
        )
        self.assertEqual(node.exit_status, 0)
        self.assertLess(np.linalg.norm(results["optimized_parameters"]), 0.1)


class TestScenarioFailures(unittest.TestCase):
    def test_crashing_oracle_graceful_stop(self):
        """Oracle failing mid-run (SCF crash analogue) must stop gracefully
        with finite parameters and the best point so far."""

        state = {"n": 0}

        def crashing(x):
            state["n"] += 1
            if state["n"] > 25:
                return None
            return float(np.sum(x**2))

        class _Crashing(WorkChain):
            @classmethod
            def define(cls, spec):
                super().define(spec)
                spec.input("x", valid_type=List)
                spec.outline(cls.run_calc, cls.finalize)
                spec.output("value", valid_type=Float)

            def run_calc(self):
                value = crashing(np.array(self.inputs.x.get_list()))
                if value is None:
                    self.ctx.result = extractor.get_penalty()
                else:
                    self.ctx.result = value

            def finalize(self):
                self.out("value", Float(self.ctx.result).store())

        class _Evaluator(EvalWorkChainProblem):
            problem_workchain = _Crashing

        results, node = run_optimizer(_Evaluator, [0.5, 0.5], dict(DEFAULTS))
        self.assertIn(node.exit_status, (0, 496))
        final = np.array(results["optimized_parameters"])
        self.assertTrue(np.all(np.isfinite(final)))
        self.assertLess(np.linalg.norm(final), 1.0)

    def test_nan_oracle_graceful_stop(self):
        """NaN values from the oracle must not leak into parameters or stop
        the run with a poisoned noise estimate."""

        def nan_inside(x):
            if np.linalg.norm(x) < 0.4:
                return float("nan")
            return float(np.sum(x**2))

        class _NanSafe(WorkChain):
            @classmethod
            def define(cls, spec):
                super().define(spec)
                spec.input("x", valid_type=List)
                spec.outline(cls.run_calc, cls.finalize)
                spec.output("value", valid_type=Float)

            def run_calc(self):
                value = nan_inside(np.array(self.inputs.x.get_list()))
                self.ctx.result = value if np.isfinite(value) else extractor.get_penalty()

            def finalize(self):
                self.out("value", Float(self.ctx.result).store())

        class _Evaluator(EvalWorkChainProblem):
            problem_workchain = _NanSafe

        results, node = run_optimizer(_Evaluator, [0.5, 0.5], dict(DEFAULTS, tolerance=1e-8))
        final = np.array(results["optimized_parameters"])
        self.assertTrue(np.all(np.isfinite(final)))

    def test_flat_objective_preserves_parameters(self):
        results, node = run_optimizer(
            make_oracle(lambda x: 42.0),
            [1.0, 1.0],
            dict(DEFAULTS, tolerance=1e-8),
        )
        self.assertIn(node.exit_status, (0, 496))
        np.testing.assert_allclose(results["optimized_parameters"], [1.0, 1.0], atol=1e-8)


class TestScenarioClamping(unittest.TestCase):
    def test_clamped_ill_conditioned_monotone(self):
        """Active max_step clamp on a cond=1e5 rotated quadratic: the old
        clamp-after-Armijo bug produced 100x energy jumps here; with the
        clamp inside the line search the descent must stay monotone."""

        q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(4, 4)))
        a = q @ np.diag([1e4, 1e3, 1.0, 0.1]) @ q.T
        x0 = np.random.default_rng(0).normal(size=4) * 0.5
        f_start = 0.5 * x0 @ a @ x0

        def objective(x):
            return float(0.5 * x @ a @ x)

        results, node = run_optimizer(
            make_oracle(objective),
            list(x0),
            dict(DEFAULTS, tolerance=1e-7, max_step=0.1) | {"itmax": 120},
        )
        self.assertEqual(node.exit_status, 0)
        values = [entry["value"] for entry in results["history"]]
        for previous, current in zip(values, values[1:], strict=False):
            self.assertLessEqual(current, previous + 1e-9)
        f_final = 0.5 * np.array(results["optimized_parameters"]) @ a @ np.array(results["optimized_parameters"])
        self.assertLess(f_final, 1e-3 * f_start)

    def test_deterministic_replay_identical(self):
        """Two runs of the same deterministic oracle must be identical."""

        def make_counter_oracle():
            state = {"n": 0}

            def objective(x):
                state["n"] += 1
                return float(np.sum(x**2) + 1e-9 * np.sin(state["n"]))

            return objective

        first, node1 = run_optimizer(make_oracle(make_counter_oracle()), [0.4, -0.4], dict(DEFAULTS))
        second, node2 = run_optimizer(make_oracle(make_counter_oracle()), [0.4, -0.4], dict(DEFAULTS))
        self.assertEqual(node1.exit_status, node2.exit_status)
        np.testing.assert_allclose(
            np.array(first["optimized_parameters"]),
            np.array(second["optimized_parameters"]),
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
