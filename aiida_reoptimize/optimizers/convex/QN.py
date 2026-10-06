import numpy as np
from aiida.orm import List

from ..result_utils import ensure_population_has_valid_results
from .base import _GDBase


class BFGSOptimizer(_GDBase):
    """BFGS quasi-Newton optimization WorkChain.

    Uses the BFGS formula to approximate the inverse Hessian and performs
    backtracking line search with the Armijo (sufficient decrease) condition.
    """

    def initialize(self):
        exit_code = super().initialize()
        if exit_code is not None:
            return exit_code
        self.initialize_step_control()
        self.ctx.inv_hessian = np.eye(len(self.ctx.parameters))
        self.ctx.gradient_prev = None
        self.ctx.parameters_prev = None

        # Line search parameters
        self.ctx.alpha = self.inputs["parameters"].get("algorithm_settings", {}).get("alpha", 1.0)
        self.ctx.beta = self.inputs["parameters"].get("algorithm_settings", {}).get("beta", 0.5)

        self.ctx.sigma = self.inputs["parameters"].get("algorithm_settings", {}).get("sigma", 1e-4)

        self.ctx.linesearch_max_iter = (
            self.inputs["parameters"].get("algorithm_settings", {}).get("linesearch_max_iter", 20)
        )

    def _reset_after_jump(self):
        """Reset BFGS state after a random jump."""

        self.ctx.inv_hessian = np.eye(len(self.ctx.parameters))
        self.ctx.gradient_prev = None
        self.ctx.parameters_prev = None

    def _line_search(self, direction):
        """Backtracking line search with Armijo (sufficient decrease) condition.

        Iteratively reduces the step size by ``beta`` until the Armijo condition
        is satisfied, ensuring sufficient decrease in the objective function.

        Args:
            direction: Search direction vector.

        Returns:
            Step size (float), or an exit code if evaluation fails.
        """
        alpha = self.ctx.alpha
        beta = self.ctx.beta
        sigma = self.ctx.sigma
        max_iter = self.ctx.linesearch_max_iter

        params = self.ctx.parameters
        f0 = self.ctx.results[0]
        grad = self.ctx.gradient

        for _ in range(max_iter):
            self.report(f"Performing line search iteration {_ + 1}")
            trial_params = params + alpha * direction
            trial_targets = [trial_params.tolist()]
            raw_trial_results = self.run_evaluator(
                List(trial_targets),
                calculator_parameters=self.ctx.calculator_parameters,
            )
            if raw_trial_results is None:
                return self.exit_codes.ERROR_EVALUATOR_FAILED
            extracted_trial_results = self.extractor(raw_trial_results["evaluation_results"])
            exit_code = ensure_population_has_valid_results(
                self,
                extracted_trial_results,
                raw_results=raw_trial_results["evaluation_results"],
                context=f"line search iteration {_ + 1}",
            )
            if exit_code is not None:
                return exit_code
            f_trial = extracted_trial_results[0]
            if f_trial <= f0 + sigma * alpha * np.dot(grad, direction):
                return alpha
            alpha *= beta
        self.report(
            f"Line search failed to find a suitable step size, using alpha={self.ctx.alpha * 1e-3}"  # noqa: E501
        )
        return self.ctx.alpha * 1e-3

    def update_parameters(self, gradient: np.ndarray):
        """Update parameters using BFGS direction and step size."""
        exit_code = self.handle_worse_objective(
            rate_key="alpha",
            on_jump=self._reset_after_jump,
        )
        if exit_code is not None:
            return exit_code

        if self.ctx.iteration == 1:
            direction = -np.dot(self.ctx.inv_hessian, gradient)
        else:
            s = self.ctx.parameters - self.ctx.parameters_prev
            y = gradient - self.ctx.gradient_prev
            ys = np.dot(y, s)
            if ys > self.ctx.epsilon:
                I = np.eye(len(self.ctx.parameters))  # noqa: E741
                rho = 1.0 / ys
                V = I - rho * np.outer(s, y)
                self.ctx.inv_hessian = V @ self.ctx.inv_hessian @ V.T + rho * np.outer(s, s)
            direction = -np.dot(self.ctx.inv_hessian, gradient)

        step_size = self._line_search(direction)
        if hasattr(step_size, "status") and step_size.status != 0:
            return step_size

        self.ctx.parameters_prev = self.ctx.parameters.copy()
        self.ctx.gradient_prev = gradient.copy()
        step = step_size * direction
        step = self.clamp_step(step)

        self.record_history(
            parameters=self.ctx.parameters,
            gradient=gradient,
            value=self.ctx.results[0],
            step_rate=step_size,
            step=step,
        )

        self.report_progress()
        self.ctx.parameters = self.ctx.parameters + step
        self.ctx.iteration += 1


class LBFGSOptimizer(_GDBase):
    """Limited-memory BFGS optimization WorkChain.

    Deterministic quasi-Newton method designed for smooth objectives with an
    expensive deterministic oracle (e.g. full SCF cascades), robust to a
    moderate amount of oracle noise (e.g. numerical SCF drift). Unlike the
    Adam/RMSprop optimizers, no moment statistics of gradient noise are
    tracked: the inverse Hessian is approximated from curvature (secant)
    pairs, and every step is accepted through a backtracking Armijo line
    search, which guarantees monotone descent without rollbacks or random
    jumps.

    Algorithm (Nocedal & Wright, ch. 7, with the stochastic safeguards of
    Berahas, Byrd & Nocedal, SIAM J. Optim. 26/4, 2016):
        1. direction d = -H_k g via the two-loop recursion over the last
           ``memory`` secant pairs (s_k = x_k - x_{k-1}, y_k = g_k - g_{k-1});
        2. step size alpha from a backtracking line search with the Armijo
           (sufficient decrease) condition, relaxed by a noise allowance
           proportional to the estimated oracle standard deviation;
        3. the accepted step always yields a secant pair (curvature
           information is never discarded), unless the pair is dominated
           by gradient noise;
        4. when the gradient norm reaches the noise floor of the
           finite-difference estimator, the optimization stops with an
           explicit noise-floor report instead of looping forever.

    Noise handling:
        - the oracle standard deviation sigma_f is estimated from repeated
          evaluations of the same point: one duplicated probe before the
          first step, then one additional evaluation of every accepted line
          search point (reusing the value the line search already computed),
          accumulated as an exponentially weighted moving average so a
          single unlucky probe cannot leave the noise level undetected;
        - non-finite and penalty values are filtered out of the noise
          estimate: a single poisoned evaluation must not corrupt the
          acceptance threshold or the noise floor;
        - secant pairs whose curvature signal is comparable to the noise
          level are rejected, so noise cannot corrupt the Hessian
          approximation;
        - the convergence criterion |g| < tolerance is superseded by the
          noise floor |g| < sqrt(n) * sigma_g when the latter is larger,
          where sigma_g = sigma_f / delta is the noise floor of the
          forward-difference gradient.
    """

    def initialize(self):
        exit_code = super().initialize()
        if exit_code is not None:
            return exit_code
        self.ctx.s_list = []
        self.ctx.y_list = []
        self.ctx.gradient_prev = None
        self.ctx.parameters_prev = None

        settings = self.inputs["parameters"].get("algorithm_settings", {})
        self.ctx.memory = settings.get("memory", 10)
        self.ctx.c1 = settings.get("c1", 1e-4)
        self.ctx.c2 = settings.get("c2", 0.9)
        self.ctx.max_linesearch_iter = settings.get("linesearch_max_iter", 20)
        self.ctx.alpha_init = settings.get("alpha_init", 1.0)
        self.ctx.alpha_min = settings.get("alpha_min", 1e-12)
        self.ctx.curvature_factor = settings.get("curvature_factor", 1e-12)
        self.ctx.noise_ewma = settings.get("noise_ewma", 0.9)
        self.ctx.noise_allowance = settings.get("noise_allowance", 2.0)
        self.ctx.noise_floor_factor = settings.get("noise_floor_factor", 2.0)
        self.ctx.patience = settings.get("patience", 5)
        self.ctx.sigma_f = None
        self.ctx.best_value = None
        self.ctx.stall_count = 0

    def _evaluate_targets(self, targets):
        """Evaluate a batch of targets, returning values or an exit code."""

        if self.ctx.calculator_parameters:
            raw_results = self.run_evaluator(
                List(targets),
                calculator_parameters=self.ctx.calculator_parameters,
            )
        else:
            raw_results = self.run_evaluator(List(targets))
        if raw_results is None:
            return self.exit_codes.ERROR_EVALUATOR_FAILED
        raw_results = raw_results["evaluation_results"]
        results = self.extractor(raw_results)
        exit_code = ensure_population_has_valid_results(
            self,
            results,
            raw_results=raw_results,
            context="line search",
        )
        if exit_code is not None:
            return exit_code
        return results

    def _noise_allowance(self):
        """Return the noise allowance for acceptance thresholds (0 when the
        estimate is missing or unusable)."""

        sigma = self.ctx.sigma_f
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            return 0.0
        return self.ctx.noise_allowance * sigma

    def _evaluate_probe(self, targets):
        """Evaluate a batch of noise-probe targets without the
        all-penalty abort.

        Noise probes are auxiliary: a failed (penalty) evaluation simply
        yields no sample rather than aborting the optimization. The main
        loop keeps its stricter ``ensure_population_has_valid_results``
        semantics for optimization targets.

        Returns:
            List of extracted values (may contain penalties), or ``None``
            on evaluator failure.
        """

        if self.ctx.calculator_parameters:
            raw_results = self.run_evaluator(
                List(targets),
                calculator_parameters=self.ctx.calculator_parameters,
            )
        else:
            raw_results = self.run_evaluator(List(targets))
        if raw_results is None:
            return None
        return self.extractor(raw_results["evaluation_results"])

    def _fold_noise_sample(self, f_known, target):
        """Refine the noise estimate from ONE extra evaluation of ``target``.

        The line search has already evaluated ``target`` and produced
        ``f_known``; a single additional evaluation of the same point is
        enough for a duplicated-probe difference, so the refresh costs one
        oracle run instead of two.

        Non-finite values and extractor penalties are filtered out: a
        single poisoned evaluation must not corrupt sigma_f (a NaN would
        poison the Armijo allowance, a penalty value would inflate the
        noise floor by orders of magnitude).

        Args:
            f_known: Value already evaluated at ``target``.
            target: Parameter vector to re-evaluate once.

        Returns:
            The new sample pair when a valid estimate was folded in,
            ``None`` otherwise (evaluation failure or filtered values).
        """

        results = self._evaluate_probe([target])
        if results is None:
            return None
        f_again = results[0]
        penalty = self.extractor.get_penalty()
        for value in (f_known, f_again):
            if not np.isfinite(value) or value == penalty:
                return None
        self._accumulate_noise_sample(f_known, f_again)
        return f_known, f_again

    def _accumulate_noise_sample(self, f_first, f_second):
        """Fold a duplicated-probe difference into the EWMA noise estimate.

        A single duplicated probe can underestimate the noise by orders
        of magnitude (two nearly identical draws); continuous re-estimation
        from every accepted step keeps sigma_f honest.
        """

        sample_var = 0.5 * (f_first - f_second) ** 2
        if self.ctx.sigma_f is None or not np.isfinite(self.ctx.sigma_f):
            self.ctx.sigma_f = np.sqrt(sample_var)
        else:
            alpha_ewma = self.ctx.noise_ewma
            old_var = self.ctx.sigma_f**2
            self.ctx.sigma_f = np.sqrt(alpha_ewma * old_var + (1.0 - alpha_ewma) * sample_var)

    def _seed_noise_estimate(self, target):
        """Seed the noise estimate with a duplicated probe of ``target``.

        Returns:
            ``True`` when a valid estimate was folded in.
        """

        results = self._evaluate_probe([target, target])
        if results is None:
            return False
        f_first, f_second = results[0], results[1]
        penalty = self.extractor.get_penalty()
        if not (np.isfinite(f_first) and np.isfinite(f_second)):
            return False
        if f_first == penalty or f_second == penalty:
            return False
        self._accumulate_noise_sample(f_first, f_second)
        return True

    def _noise_floor(self, gradient):
        """Return the gradient-norm noise floor, or None when noiseless.

        The forward-difference gradient amplifies the oracle noise by
        1/delta per component, so the expected norm of the noise-driven
        gradient is approximately sqrt(n) * sigma_f / delta.
        """

        if self.ctx.sigma_f is None or self.ctx.sigma_f <= 0 or self.ctx.delta <= 0:
            return None
        return self.ctx.noise_floor_factor * np.sqrt(len(gradient)) * self.ctx.sigma_f / self.ctx.delta

    def _two_loop(self, gradient):
        """Compute the L-BFGS direction -H g via the two-loop recursion."""

        q = gradient.copy()
        alphas = []
        for s, y in reversed(list(zip(self.ctx.s_list, self.ctx.y_list, strict=True))):
            rho = 1.0 / np.dot(y, s)
            alpha = rho * np.dot(s, q)
            alphas.append(alpha)
            q = q - alpha * y

        if self.ctx.s_list:
            s_last, y_last = self.ctx.s_list[-1], self.ctx.y_list[-1]
            gamma = np.dot(y_last, s_last) / np.dot(y_last, y_last)
            r = gamma * q
        else:
            r = q

        pairs = list(zip(self.ctx.s_list, self.ctx.y_list, strict=True))
        for alpha, (s, y) in zip(reversed(alphas), pairs, strict=True):
            rho = 1.0 / np.dot(y, s)
            beta = rho * np.dot(y, r)
            r = r + (alpha - beta) * s
        return -r

    def _linesearch(self, params, value, gradient, direction, first_step):
        """Backtracking line search with the Armijo (sufficient decrease)
        condition under possible oracle noise.

        Each trial costs a single oracle evaluation (the objective value at
        the trial point). The full finite-difference gradient at the accepted
        trial point is recomputed once in the next outer iteration by the
        standard ``optimization_process`` loop, so no extra probes are
        spent inside the line search.

        Returns the accepted step size, the accepted (clamped) point, and
        the evaluated value at that point; the value is reused by the
        caller for the noise refresh without a second evaluation.

        Failed (penalty) trial evaluations are treated as an infinite
        objective value: the Armijo condition rejects them and the search
        backtracks, instead of aborting the whole run.

        The sufficient decrease condition is relaxed by
        ``noise_allowance * sigma_f`` (0 while the oracle appears
        noiseless), so oracle noise neither rejects good steps nor
        accepts bad ones (Berahas, Byrd & Nocedal, 2016).

        Trial candidates are clamped BEFORE evaluation: the accepted point
        is always the exact point that satisfied the Armijo condition.
        Clamping after acceptance would move the optimizer to an
        unevaluated point and break the guaranteed descent.

        On the first step the initial step length is scaled to the gradient
        norm (Nocedal & Wright, eq. 7.20 alternative initialization): with a
        unit step, an ill-conditioned gradient overshoots the minimum along
        the stiff directions, and the resulting secant pairs collapse onto a
        low-dimensional subspace.
        """

        if first_step:
            grad_norm = np.linalg.norm(gradient)
            alpha = min(self.ctx.alpha_init, 1.0 / grad_norm) if grad_norm > 0 else self.ctx.alpha_init
        else:
            alpha = self.ctx.alpha_init
        gTd = np.dot(gradient, direction)
        allowance = self._noise_allowance()
        penalty = self.extractor.get_penalty()
        best = None
        for _ in range(self.ctx.max_linesearch_iter):
            trial = params + alpha * direction
            trial = self.clamp_parameters(trial, params)
            results = self._evaluate_probe([trial.tolist()])
            if results is None:
                return None, None, None
            f_trial = results[0]
            if f_trial == penalty or not np.isfinite(f_trial):
                alpha *= 0.5
                if alpha < self.ctx.alpha_min:
                    break
                continue
            if f_trial <= value + self.ctx.c1 * alpha * gTd + allowance:
                best = (alpha, trial, f_trial)
                break
            alpha *= 0.5
            if alpha < self.ctx.alpha_min:
                break
        if best is None:
            return None, None, None
        return best

    def clamp_parameters(self, trial, previous):
        """Clamp a candidate point component-wise to stay within
        ``max_step`` of the previous point.

        Called from the line search on every candidate BEFORE evaluation,
        so the accepted point is always the exact evaluated point and the
        Armijo guarantee applies to the realized move.
        """

        if not self.ctx.max_step:
            return trial
        clamped = np.clip(trial, previous - self.ctx.max_step, previous + self.ctx.max_step)
        if not np.allclose(clamped, trial):
            self.report(f"Step was clamped to +/-{self.ctx.max_step} per component.")
        return clamped

    def _update_stall_tracking(self, value):
        """Track the best observed value and count noise-level stalls.

        A value improves the best only when it is better by more than the
        noise allowance; once ``patience`` consecutive accepted steps yield
        no significant improvement, the objective is at its noise floor and
        the caller should stop.

        Returns:
            True when the optimization should stop due to stagnation.
        """

        allowance = self._noise_allowance()
        if self.ctx.best_value is None or value < self.ctx.best_value - allowance:
            self.ctx.best_value = value
            self.ctx.stall_count = 0
            return False
        self.ctx.stall_count += 1
        return self.ctx.stall_count >= self.ctx.patience

    def update_parameters(self, gradient):
        """Perform one L-BFGS step: two-loop direction + Armijo line search."""

        floor = self._noise_floor(gradient)
        if floor is not None and np.linalg.norm(gradient) <= floor:
            self.report(
                f"Gradient norm {np.linalg.norm(gradient):.3e} is at the estimated "
                f"noise floor {floor:.3e}; stopping at the noise floor."
            )
            self.ctx.converged = True
            return

        if self.ctx.sigma_f is None:
            if self._seed_noise_estimate(self.ctx.parameters.tolist()):
                self.report(f"Estimated oracle noise sigma_f = {self.ctx.sigma_f:.3e}")
            else:
                self.report("Noise seeding probe failed; assuming a noiseless oracle.")

        direction = self._two_loop(gradient)
        alpha, trial, f_trial = self._linesearch(
            self.ctx.parameters,
            self.ctx.results[0],
            gradient,
            direction,
            first_step=not self.ctx.s_list,
        )
        if alpha is None:
            self.report("Line search failed to find an acceptable step size.")
            self.ctx.converged = True
            return

        if self._fold_noise_sample(f_trial, trial.tolist()):
            self.report(f"Oracle noise estimate sigma_f = {self.ctx.sigma_f:.3e}")

        step = alpha * direction

        if self._update_stall_tracking(self.ctx.results[0]):
            self.report(
                f"No significant improvement for {self.ctx.patience} steps "
                f"(best value {self.ctx.best_value:.6e}); stopping at the noise floor."
            )
            self.ctx.converged = True
            return

        self.record_history(
            parameters=self.ctx.parameters,
            gradient=gradient,
            value=self.ctx.results[0],
            step_rate=alpha,
            step=step,
        )
        self.report_progress()

        if self.ctx.parameters_prev is not None:
            s = self.ctx.parameters - self.ctx.parameters_prev
            y = gradient - self.ctx.gradient_prev
            ys = np.dot(y, s)
            if ys > self.ctx.curvature_factor * np.linalg.norm(y) * np.linalg.norm(s):
                self.ctx.s_list.append(s)
                self.ctx.y_list.append(y)
                if len(self.ctx.s_list) > self.ctx.memory:
                    self.ctx.s_list.pop(0)
                    self.ctx.y_list.pop(0)
            else:
                self.report("Secant pair rejected: curvature signal at noise level.")

        self.ctx.parameters_prev = self.ctx.parameters.copy()
        self.ctx.gradient_prev = gradient.copy()
        self.ctx.parameters = trial
        self.ctx.iteration += 1
