from aiida.engine import WorkChain, ToContext, while_
from aiida.orm import Float, List, Int, Str, Bool
import numpy as np

class BFGSWorkChain(WorkChain):
    """
    BFGS WorkChain for parameter optimization using a target function.
    The target function is expected to be a callable (awitable) object.
    Implements numerical gradient calculation and Hessian approximation using BFGS update.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('initial_parameters', valid_type=List, help='Начальные параметры для оптимизации.')
        spec.input('tolerance', valid_type=Float, default=lambda: Float(1e-6), help='Критерий остановки по изменению параметров.')
        spec.input('max_iterations', valid_type=Int, default=lambda: Int(100), help='Максимальное количество итераций.')
        spec.input('key_value', valid_type=Str, help='Ключ оптимизируемого значения')

        spec.outline(
            cls.initialize,
            while_(cls.should_continue)(
                cls.update_hessian, # Обновляет параметры + делает новый гессиан
                cls.generate_targets, # Записывает точки, которые нужно посчитать для расчета градиента
                cls.evaluate, # считает целевую функию и значения для градиента
                cls.collect_results, # записывает значения в контекст
                cls.evaluate_gradient_numerically, # вычисляет градиент
                cls.update_parameters,
            ),
            cls.finalize
        )

        spec.output('optimized_parameters', valid_type=List, required=True, help='Оптимизированные параметры.')
        spec.output('final_value', valid_type=Float, required=True, help='Значение целевой функции в оптимуме.')

    def initialize(self):
        """Initialize context variables and optimization parameters."""
        self.ctx.parameters = np.array(self.inputs.initial_parameters)
        self.ctx.inv_hessian = np.eye(len(self.ctx.parameters))  # Начальное приближение обратной матрицы Гессе
        self.ctx.iteration = 0
        self.ctx.tolerance = self.inputs.tolerance.value
        self.ctx.max_iterations = self.inputs.max_iterations
        self.ctx.converged = False
        self.ctx.gradient = None
        # RMSprop
        self.ctx.learning_rate = 0.001
        self.ctx.rho = 0.9
        self.ctx.epsilon = 1e-8
        self.ctx.accumulated_grad_sq = 0

    def should_continue(self):
        """Проверка условия продолжения итераций."""
        return Bool(not self.ctx.converged and self.ctx.iteration < self.inputs.max_iterations.value)

    def generate_targets(self):
        """Generate parameter sets for gradient calculation.
        
        Creates base parameters and parameters with small perturbations (EPSILON)
        for finite difference gradient estimation.
        """

        params = self.ctx.parameters.tolist()

        targets = []
        targets.append(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += 1e-5 # !XXX Epsilon
            targets.append(params_plus)

        self.ctx.targets = targets

    def evaluate(self):
        """Evaluate the target function and gradient points.
        
        Must be implemented by subclasses to execute actual function evaluations.
        Expected to store results in self.ctx[f'eval_{i}'] for each target.
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def collect_results(self):
        """Collect evaluation results from sub-processes.
        
        Handles failed evaluations by assigning infinite penalty value.
        Stores results in self.ctx.results for gradient calculation.
        """
        results = []
        for i in range(len(self.ctx.targets)):
            process = self.ctx[f'eval_{i}']
            if process.is_finished_ok:
                res = process.outputs[self.inputs.key_value.value].value
            else:
                res = np.inf  # Штраф за ошибку
                self.report(f"Evaluation {i} failed: {process.exit_status}")
            results.append(res)
        self.ctx.results = results

    def evaluate_gradient_numerically(self):
        """Compute numerical gradient using finite differences.
        
        Uses stored function evaluations to calculate gradient components.
        Updates convergence status based on gradient norm.
        """
        func_value = self.ctx.results[0]
        gradient_targets = self.ctx.results[1:]

        gradient = []

        for f_plus in gradient_targets:
            grad_i = (f_plus - func_value) / 1e-5 # !XXX Epsilon
            gradient.append(grad_i)

        if getattr(self.ctx, 'gradient', None) is not None:
            self.ctx.gradient_prev = self.ctx.gradient

        self.ctx.gradient = np.array(gradient)
        current_gradient_norm = np.linalg.norm(self.ctx.gradient) 

        self.report(f"Iteration: {self.ctx.iteration}; Current gradient norm. value is {current_gradient_norm}")

        if current_gradient_norm < self.inputs.tolerance.value:
            self.ctx.converged = True

    def update_parameters(self):
        """Update parameters using BFGS direction and step size."""

        search_direction = -np.dot(self.ctx.inv_hessian, self.ctx.gradient)
        
        step_size = self._update_RMSprop()

        self.report(f"Current step size is {step_size}")
        self.ctx.parameters_prev = self.ctx.parameters
        self.ctx.parameters = self.ctx.parameters + step_size * search_direction

    def _update_RMSprop(self):
        """Compute RMSprop adaptive step size.
        
        Updates squared gradient accumulator and calculates adaptive step.
        """
        self.ctx.accumulated_grad_sq = self.ctx.rho * self.ctx.accumulated_grad_sq + (1 - self.ctx.rho) * self.ctx.gradient**2
        return self.ctx.learning_rate / np.sqrt(self.ctx.accumulated_grad_sq + self.ctx.epsilon) * self.ctx.gradient

    def update_hessian(self):
        """Update inverse Hessian approximation using BFGS formula.
        
        Skipped for first iteration (identity matrix used initially).
        """
        if self.ctx.iteration > 1:
            s = self.ctx.parameters - self.ctx.parameters_prev
            y = self.ctx.gradient - self.ctx.gradient_prev
            rho = 1.0 / np.dot(y, s)

            I = np.eye(len(self.ctx.parameters))
            self.ctx.inv_hessian = (
                I - rho * np.outer(s, y)
            ).dot(self.ctx.inv_hessian).dot(
                I - rho * np.outer(y, s)
            ) + rho * np.outer(s, s)
        else:
            pass

        self.ctx.iteration += 1

    def finalize(self):
        """Store final results in AiiDA outputs.
        
        Saves optimized parameters and final function value.
        Handles cases where final evaluation failed.
        """
        self.out('optimized_parameters', List(list=self.ctx.parameters.tolist()).store())

        if hasattr(self.ctx, 'results') and len(self.ctx.results) > 0:
            self.out('final_value', Float(self.ctx.results[0]).store())
        else:
            self.report("Предупреждение: финальное значение не было рассчитано")
            self.out('final_value', Float(0).store())