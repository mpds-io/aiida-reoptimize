from aiida.engine import WorkChain, while_
from aiida.orm import Float, List, Int, Str, Bool
import numpy as np


class _SDGBase(WorkChain):
    """Basis for SDG based optimization algorithm"""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('initial_parameters', valid_type=List, help='Начальные параметры для оптимизации.')
        spec.input('tolerance', valid_type=Float, default=lambda: Float(1e-3))
        spec.input('max_iterations', valid_type=Int, default=lambda: Int(100))
        spec.input('key_value', valid_type=Str, help='Ключ оптимизируемого значения')
        spec.input('epsilon', valid_type=Float, default=lambda: Float(1e-8))

        spec.outline(
            cls.initialize,
            while_(cls.should_continue)(
                cls.generate_targets,
                cls.evaluate,
                cls.collect_results,
                cls.evaluate_gradient_numerically,
                cls.update_parameters,
            ),
            cls.finalize
        )

        spec.output('optimized_parameters', valid_type=List, required=True)
        spec.output('final_value', valid_type=Float, required=True)

    def initialize(self):
        self.ctx.parameters = np.array(self.inputs.initial_parameters)
        self.ctx.iteration = 1
        self.ctx.converged = False
        self.ctx.gradient = None
        
        # Инициализация для RMSprop
        self.ctx.accumulated_grad_sq = np.zeros_like(self.ctx.parameters)

    def should_continue(self):
        return Bool(not self.ctx.converged and self.ctx.iteration < self.inputs.max_iterations.value)

    def generate_targets(self):
        params = self.ctx.parameters.tolist()
        targets = [params]
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += 1e-5
            targets.append(params_plus)
        self.ctx.targets = targets

    def evaluate(self):
        raise NotImplementedError("Subclasses must implement evaluate()")

    def collect_results(self):
        results = []
        for i in range(len(self.ctx.targets)):
            process = self.ctx[f'eval_{i}']
            results.append(process.outputs[self.inputs.key_value.value].value if process.is_finished_ok else np.inf)
        self.ctx.results = results

    def evaluate_gradient_numerically(self):
        func_value = self.ctx.results[0]
        gradient = [(self.ctx.results[i+1] - func_value) / 1e-5 for i in range(len(self.ctx.parameters))]
        self.ctx.gradient = np.array(gradient)
        
        if np.linalg.norm(self.ctx.gradient) < self.inputs.tolerance.value:
            self.ctx.converged = True

    def update_parameters(self):
        raise NotImplementedError("Subclasses must implement update_parameters()")

    def finalize(self):
        self.out('optimized_parameters', List(list=self.ctx.parameters.tolist()).store())
        self.out('final_value', Float(self.ctx.results[0]).store())

class RMSpropWorkChain(_SDGBase):
    """
    RMSprop workchain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('learning_rate', valid_type=Float, default=lambda: Float(0.001))
        spec.input('rho', valid_type=Float, default=lambda: Float(0.9))

    def initialize(self):
        super().initialize()
        self.ctx.accumulated_grad_sq = np.zeros_like(self.ctx.parameters)

    def update_parameters(self):
        # ! XXX add np.inf / np.nan safety check
        self.ctx.accumulated_grad_sq = (
            self.inputs.rho.value * self.ctx.accumulated_grad_sq 
            + (1 - self.inputs.rho.value) * self.ctx.gradient**2
        )
        
        step = (
            self.inputs.learning_rate.value 
            / np.sqrt(self.ctx.accumulated_grad_sq + self.inputs.epsilon.value) 
            * self.ctx.gradient
        )
        self.report(f"\nIteration {self.ctx.iteration}, \n current parameters {self.ctx.parameters} \n current gradient {self.ctx.gradient}\ncurrent value {self.ctx.results[0]}")
        self.ctx.parameters -= step


class ADAMpropWorkChain(_SDGBase):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        # ! XXX Add information about parameters / test paramiters to find optimal
        spec.input('learning_rate', valid_type=Float, default=lambda: Float(0.001))
        spec.input('beta1', valid_type=Float, default=lambda: Float(0.9))
        spec.input('beta2', valid_type=Float, default=lambda: Float(0.999))

    def initialize(self):
        super().initialize()
        self.ctx.m = np.zeros_like(self.ctx.parameters)
        self.ctx.v = np.zeros_like(self.ctx.parameters)


    def update_parameters(self):
        if np.any(np.isnan(self.ctx.gradient)) or np.any(np.isinf(self.ctx.gradient)):
            self.report("Aborting: Invalid gradient (NaN/Inf detected).")
            self.ctx.converged = True
            return

        self.ctx.m = self.inputs.beta1.value * self.ctx.m + (1 - self.inputs.beta1.value) * self.ctx.gradient
        self.ctx.v = self.inputs.beta2.value * self.ctx.v + (1 - self.inputs.beta2.value) * (self.ctx.gradient ** 2)

        m_hat = self.ctx.m / (1 - self.inputs.beta1.value ** self.ctx.iteration)
        v_hat = self.ctx.v / (1 - self.inputs.beta2.value ** self.ctx.iteration)

        denominator = np.sqrt(v_hat) + self.inputs.epsilon.value
        denominator[denominator <= 0] = self.inputs.epsilon.value  # Гарантируем положительность

        step = self.inputs.learning_rate.value * m_hat / denominator

        if np.any(np.isnan(step)) or np.any(np.isinf(step)):
            self.report("Aborting: Invalid step (NaN/Inf detected).")
            self.ctx.converged = True
            return

        self.report(f"\nIteration {self.ctx.iteration}, \n current parameters {self.ctx.parameters} \n current gradient {self.ctx.gradient}\ncurrent value {self.ctx.results[0]}")
        self.ctx.parameters -= step
        self.ctx.iteration += 1
