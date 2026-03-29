from argparse import ArgumentParser

import numpy as np
from aiida import load_profile
from aiida.engine import WorkChain, run_get_node
from aiida.orm import Dict, Float, Int, List

from aiida_reoptimize.base.Evaluation import EvalWorkChainProblem
from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.optimizers.convex.GD import (
    AdamOptimizer,
    ConjugateGradientOptimizer,
    RMSpropOptimizer,
)
from aiida_reoptimize.optimizers.convex.QN import BFGSOptimizer

load_profile()


class QuadraticProblem(WorkChain):
    """Simple dummy problem: minimize f(x) = x^2 for x in R."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("x", valid_type=List)
        spec.outline(cls.evaluate)
        spec.output("value", valid_type=Float)

    def evaluate(self):
        x = np.array(self.inputs.x.get_list(), dtype=float)
        value = float(np.sum(x**2))
        self.out("value", Float(value).store())


class QuadraticEvaluator(EvalWorkChainProblem):
    problem_workchain = QuadraticProblem


extractor = BasicExtractor(node_extractor=lambda outputs: outputs["value"])


class AdamQuadraticOptimizer(AdamOptimizer):
    evaluator_workchain = QuadraticEvaluator
    extractor = extractor


class RMSpropQuadraticOptimizer(RMSpropOptimizer):
    evaluator_workchain = QuadraticEvaluator
    extractor = extractor


class CGDQuadraticOptimizer(ConjugateGradientOptimizer):
    evaluator_workchain = QuadraticEvaluator
    extractor = extractor


class BFGSQuadraticOptimizer(BFGSOptimizer):
    evaluator_workchain = QuadraticEvaluator
    extractor = extractor


def run_optimizer(name: str, optimizer_cls, settings: dict, initial: list[float], itmax: int):
    inputs = {
        "itmax": Int(itmax),
        "parameters": Dict(
            dict={
                "algorithm_settings": settings,
                "initial_parameters": List(list=initial),
            }
        ),
    }

    result, node = run_get_node(optimizer_cls, **inputs)

    if not node.is_finished_ok:
        message = f"[{name:7}] FAILED: exit_status={node.exit_status}, exit_message={node.exit_message}"
        print(message)
        return {
            "name": name,
            "ok": False,
            "x_norm": float("inf"),
            "f_opt": float("inf"),
            "x_vec": None,
            "message": message,
            "pk": node.pk,
        }

    x_opt = result["optimized_parameters"].get_list()
    f_opt = result["final_value"].value
    x_norm = float(np.linalg.norm(np.array(x_opt, dtype=float), ord=2))

    print(f"[{name:7}] |x*|_2={x_norm:.8e}, f(x*)={f_opt:.8e}, x*={x_opt}")
    return {
        "name": name,
        "ok": True,
        "x_norm": x_norm,
        "f_opt": f_opt,
        "x_vec": x_opt,
        "message": None,
        "pk": node.pk,
    }


def parse_args():
    parser = ArgumentParser(description="Validate all convex optimizers on f(x)=sum(x_i^2).")
    parser.add_argument("--dimensions", type=int, default=1, help="Problem dimensionality.")
    parser.add_argument("--initial", type=float, default=2.0, help="Initial absolute value for each dimension.")
    parser.add_argument("--itmax", type=int, default=40, help="Maximum iterations for each optimizer.")
    parser.add_argument("--tolerance-x", type=float, default=5e-2, help="Tolerance for ||x*||_2.")
    parser.add_argument("--tolerance-f", type=float, default=1e-3, help="Tolerance for f(x*).")
    parser.add_argument(
        "--algorithm-tolerance",
        type=float,
        default=1e-3,
        help="Gradient-norm tolerance used by optimization algorithms.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-4,
        help="Finite-difference step for numerical gradient.",
    )
    parser.add_argument(
        "--allow-jumps",
        action="store_true",
        help="Allow random jump escapes for optimizers that support them.",
    )
    parser.add_argument(
        "--only",
        choices=["adam", "rmsprop", "cgd", "bfgs"],
        default=None,
        help="Run only one optimizer instead of all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dimensions < 1:
        raise ValueError("--dimensions must be >= 1")

    initial = [args.initial] * args.dimensions

    checks = [
        (
            "Adam",
            AdamQuadraticOptimizer,
            {
                "learning_rate": 0.1,
                "beta1": 0.5,
                "beta2": 0.999,
                "tolerance": args.algorithm_tolerance,
                "delta": args.delta,
                "max_step": 0.1,
                "allowed_stuck": 6,
                "allowing_jumps": args.allow_jumps,
            },
        ),
        (
            "RMSprop",
            RMSpropQuadraticOptimizer,
            {
                "learning_rate": 0.02,
                "rho": 0.9,
                "tolerance": args.algorithm_tolerance,
                "delta": args.delta,
                "max_step": 0.05,
                "allowed_stuck": 10,
                "allowing_jumps": args.allow_jumps,
            },
        ),
        (
            "CGD",
            CGDQuadraticOptimizer,
            {
                "learning_rate": 0.1,
                "lr_increase": 1.1,
                "lr_decrease": 0.5,
                "tolerance": args.algorithm_tolerance,
                "delta": args.delta,
                "max_step": 0.1,
                "allowing_jumps": args.allow_jumps,
                "allowed_stuck": 6,
            },
        ),
        (
            "BFGS",
            BFGSQuadraticOptimizer,
            {
                "alpha": 1.0,
                "beta": 0.5,
                "sigma": 1e-4,
                "linesearch_max_iter": 20,
                "tolerance": args.algorithm_tolerance,
                "delta": args.delta,
                "max_step": 0.1,
                "allowed_stuck": 6,
                "allowing_jumps": args.allow_jumps,
            },
        ),
    ]

    if args.only is not None:
        wanted = args.only.lower()
        checks = [item for item in checks if item[0].lower() == wanted]

    failures = []

    print("Running convex optimizers on f(x)=sum(x_i^2)")
    print(
        f"Dimensions={args.dimensions}, initial={initial}, itmax={args.itmax}, "
        f"tol_x={args.tolerance_x}, tol_f={args.tolerance_f}, "
        f"algo_tol={args.algorithm_tolerance}, delta={args.delta}, "
        f"allowing_jumps={args.allow_jumps}, only={args.only or 'all'}"
    )
    print("=" * 48)
    for name, optimizer_cls, settings in checks:
        run_result = run_optimizer(
            name,
            optimizer_cls,
            settings,
            initial=initial,
            itmax=args.itmax,
        )
        if not run_result["ok"]:
            failures.append(run_result)
            continue

        if run_result["x_norm"] > args.tolerance_x or run_result["f_opt"] > args.tolerance_f:
            failures.append(run_result)

    print("=" * 48)
    if failures:
        print("Some optimizers did not reach the expected neighborhood:")
        for item in failures:
            if not item["ok"]:
                print(f"- {item['name']}: {item['message']} (pk={item['pk']})")
                continue

            print(
                f"- {item['name']}: |x|_2={item['x_norm']:.8e}, "
                f"f={item['f_opt']:.8e}, x={item['x_vec']} (pk={item['pk']})"
            )
        raise RuntimeError("Convex optimizer validation failed.")

    print("All convex optimizers reached the expected neighborhood of x=0.")


if __name__ == "__main__":
    main()
