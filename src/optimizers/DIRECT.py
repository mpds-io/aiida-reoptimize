from aiida.engine import WorkChain, ToContext, while_
from aiida.orm import ArrayData, Float, List, Int, Str
import numpy as np

class DIRECTWorkChain(WorkChain):
    """
    WorkChain реализующий алгоритм DIRECT для оптимизации функций
    """
    
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('lower_bounds', valid_type=List)
        spec.input('upper_bounds', valid_type=List)
        spec.input('key_value', valid_type=Str, default=lambda: Str('energy'))
        spec.input('max_iterations', valid_type=Int, default=lambda: Int(100))
        spec.input('epsilon', valid_type=Float, default=lambda: Float(1e-8))
        spec.input('penalty', valid_type=Float, default=lambda: Float(1e8))

        spec.outline(
            cls.initialize,
            while_(cls.continue_condition)(
                cls.select_potentially_optimal,
                cls.divide_rectangles,
                cls.evaluate,
                cls.collect_results,
                cls.update_best
            ),
            cls.finalize
        )
        
        spec.output('final_value', valid_type=Float)
        spec.output('optimized_parameters', valid_type=List)

    def initialize(self):
        """Инициализация начальных параметров"""
        self.ctx.dim = len(self.inputs.lower_bounds)
        # TODO fix
        if len(self.inputs.upper_bounds) != self.ctx.dim:
            raise ValueError("Размеры нижних и верхних границ должны совпадать с размерностью задачи")

        self.ctx.iteration = 0
        self.ctx.best_value = self.inputs.penalty.value + 1
        self.ctx.best_solution = np.zeros(self.ctx.dim)
        
        # Начальный гиперпрямоугольник
        initial_rect = ArrayData()
        lower_bounds = np.array(self.inputs.lower_bounds)
        upper_bounds = np.array(self.inputs.upper_bounds)

        initial_rect.set_array('lower', lower_bounds)
        initial_rect.set_array('upper', upper_bounds)
        initial_rect.set_array('center', (lower_bounds + upper_bounds)/2)
        initial_rect.set_array(self.inputs.key_value.value, np.array([self.inputs.penalty.value]*self.ctx.dim))
        self.ctx.rectangles = [initial_rect]

    def continue_condition(self):
        """Условие продолжения итераций"""
        return (
            self.ctx.iteration < self.inputs.max_iterations and
            np.max([np.max(r.get_array('upper') - r.get_array('lower')) 
                    for r in self.ctx.rectangles]) > self.inputs.epsilon
        )

    def select_potentially_optimal(self):
        key = self.inputs.key_value.value
        """Выбор потенциально оптимальных гиперпрямоугольников [[3]]"""
        # Сортировка по значению функции
        sorted_rects = sorted(self.ctx.rectangles, 
                            key=lambda x: x.get_array(key))
        
        po_rects = []
        for i, rect_i in enumerate(sorted_rects):
            is_optimal = True
            for j, rect_j in enumerate(sorted_rects):
                if i == j:
                    continue
                if rect_i.get_array(key) > rect_j.get_array(key) - 1e-8*abs(rect_j.get_array(key)):
                    is_optimal = False
                    break
            if is_optimal:
                po_rects.append(rect_i)
        
        self.ctx.current_rectangles = po_rects


    def divide_rectangles(self):
        """Деление выбранных гиперпрямоугольников"""
        new_rects = []
        for rect in self.ctx.current_rectangles:
            lower = rect.get_array('lower')
            upper = rect.get_array('upper')
            dim = np.argmax(upper - lower)
            delta = (upper[dim] - lower[dim])/3
            c = (lower[dim] + upper[dim])/2
            
            # Создаем 3 новых гиперпрямоугольника
            for part in range(3):
                new_lower = lower.copy()
                new_upper = upper.copy()
                
                if part == 0:
                    new_upper[dim] = c
                elif part == 1:
                    new_lower[dim] = c
                    new_upper[dim] = c + delta
                else:
                    new_lower[dim] = c - delta
                    new_upper[dim] = c
                
                new_rect = ArrayData()
                new_rect.set_array('lower', new_lower)
                new_rect.set_array('upper', new_upper)
                new_rect.set_array('center', (new_lower + new_upper)/2)
                new_rects.append(new_rect)
        
        self.ctx.new_rectangles = new_rects # list of ArrayData
        self.ctx.targets = [i.get_array('center') for i in new_rects] # чтобы не нужно было переделывать проблемы
        self.report(f"Iteration {self.ctx.iteration}: new rectangles = {self.ctx.targets}")

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
            process = self.ctx[f'eval_{i}']
            res = process.outputs[key].value if process.is_finished_ok else self.inputs.penalty.value
            
            # Create a new ArrayData node
            new_rect = ArrayData()
            new_rect.set_array('lower', rect.get_array('lower'))
            new_rect.set_array('upper', rect.get_array('upper'))
            new_rect.set_array('center', rect.get_array('center'))
            new_rect.set_array(key, np.array([res]))  # Modify before storing
            
            updated_rects.append(new_rect)
        
        # Replace old rectangles with new ones in the context
        self.ctx.new_rectangles = updated_rects

    def update_best(self):
        """Обновление лучшего решения"""
        for rect in self.ctx.new_rectangles:
            value = rect.get_array(self.inputs.key_value.value)[0]
            if value < self.ctx.best_value:
                self.ctx.best_value = value
                self.ctx.best_solution = rect.get_array('center')
        
        # Обновляем список гиперпрямоугольников
        self.ctx.rectangles = [
            r for r in self.ctx.rectangles if r not in self.ctx.current_rectangles
        ] + self.ctx.new_rectangles
        self.ctx.iteration += 1

        self.report(f"Iteration {self.ctx.iteration}: best value = {self.ctx.best_value}, best solution = {self.ctx.best_solution}")

    def finalize(self):
        """Финальные действия"""
        self.out('final_value', Float(self.ctx.best_value).store())
        self.out('optimized_parameters', List(self.ctx.best_solution.tolist()).store())