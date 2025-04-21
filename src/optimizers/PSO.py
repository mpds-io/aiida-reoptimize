import numpy as np
from aiida.engine import WorkChain
from aiida.orm import Float, Int, List, Str


class PSOWorkChain(WorkChain):
    """
    Particle Swarm Optimization WorkChain for AiiDA workflows.
    
    Implements PSO algorithm using AiiDA's workflow engine for provenance tracking.
    Requires implementation of problem-specific `evaluate()` method.
    """  # noqa: E501
    
    @classmethod
    def define(cls, spec):
        """Define input/output ports and workflow outline."""
        super().define(spec)
        
        # Input parameters with defaults
        spec.input('key_value', valid_type=Str, 
                  help='Result key identifier for sub-process outputs')
        spec.input('num_particles', valid_type=Int, default=Int(10),
                  help='Swarm size for optimization')
        spec.input('max_iterations', valid_type=Int, default=Int(5),
                  help='Maximum number of optimization cycles')
        spec.input('dimensions', valid_type=Int, default=Int(1),
                  help='Dimensionality of parameter space')
        spec.input('bounds', valid_type=List, default=List(list=[[3.7, 4.2]]),
                  help='Search space boundaries [min, max] per dimension')
        spec.input('inertia_weight', valid_type=Float, default=Float(0.7),
                  help='Velocity inertia coefficient')
        spec.input('cognitive_param', valid_type=Float, default=Float(1.5),
                  help='Cognitive learning factor')
        spec.input('social_param', valid_type=Float, default=Float(1.5),
                  help='Social learning factor')
        spec.input('penalty', valid_type=Float, default=Float(1e10),
                  help='Penalty value for failed calculations')

        # Workflow steps
        spec.outline(
            cls.initialize,
            cls.evaluate,   # Launch particle evaluations
            cls.collect_results,  # Handle results/errors
            cls.update_particles,  # PSO update logic
            cls.finalize
        )
        
        # Outputs
        spec.output('best_position', valid_type=List,
                   help='Optimal parameter set found')
        spec.output('best_value', valid_type=Float,
                   help='Best objective function value')

    def initialize(self):
        """Initialize PSO parameters and particle positions."""
        self.ctx.num_particles = self.inputs.num_particles.value
        self.ctx.iteration = 0
        self.ctx.max_iterations = self.inputs.max_iterations.value
        self.ctx.dimensions = self.inputs.dimensions.value
        self.ctx.bounds = np.array(self.inputs.bounds.get_list())
        
        # Initialize particles within bounds
        self.ctx.positions = np.random.uniform(
            self.ctx.bounds[:, 0],
            self.ctx.bounds[:, 1],
            (self.ctx.num_particles, self.ctx.dimensions)
        )
        self.ctx.velocities = np.zeros_like(self.ctx.positions)
        self.ctx.personal_best_positions = self.ctx.positions.copy()
        self.ctx.personal_best_values = np.full(self.ctx.num_particles, np.inf)
        self.ctx.global_best_value = np.inf
        self.ctx.global_best_position = self.ctx.positions[0].tolist()

    def evaluate(self):
        """
        Abstract method for particle evaluations (must be implemented).
        
        Should create sub-processes for each particle and store them in:
        self.ctx[f'eval_{i}'] for i in range(num_particles)
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def collect_results(self):
        """
        Gather evaluation results with error handling.
        
        Applies penalty values to failed calculations to maintain optimization flow.
        """  # noqa: E501
        results = []
        for i in range(self.ctx.num_particles):
            process = self.ctx[f'eval_{i}']
            if process.is_finished_ok:
                res = process.outputs[self.inputs.key_value.value].value
            else:
                res = self.inputs.penalty.value  # Apply penalty for failures
            results.append(res)
        self.ctx.results = results

    def update_particles(self):
        """
        Core PSO update logic implementing velocity/position updates.
        
        Uses: 
        - Inertia weight for velocity damping
        - Cognitive/social parameters for learning
        - Boundary enforcement with np.clip()
        """
        # Update personal bests
        for i in range(self.ctx.num_particles):
            if self.ctx.results[i] < self.ctx.personal_best_values[i]:
                self.ctx.personal_best_values[i] = self.ctx.results[i]
                self.ctx.personal_best_positions[i] = self.ctx.positions[i]

        # Update global best
        current_best_idx = np.argmin(self.ctx.personal_best_values)
        current_best_value = self.ctx.personal_best_values[current_best_idx]
        if current_best_value < self.ctx.global_best_value:
            self.ctx.global_best_value = current_best_value
            self.ctx.global_best_position = self.ctx.positions[current_best_idx].tolist()  # noqa: E501

        # Velocity update equation
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive = self.inputs.cognitive_param.value * r1 * (
            self.ctx.personal_best_positions - self.ctx.positions
        )
        social = self.inputs.social_param.value * r2 * (
            np.array(self.ctx.global_best_position) - self.ctx.positions
        )
        self.ctx.velocities = (
            self.inputs.inertia_weight.value * self.ctx.velocities
            + cognitive
            + social
        )
        
        # Position update and boundary enforcement
        self.ctx.positions += self.ctx.velocities
        self.ctx.positions = np.clip(
            self.ctx.positions,
            self.ctx.bounds[:, 0],
            self.ctx.bounds[:, 1]
        )

        # Continue iteration if needed
        if self.ctx.iteration < self.ctx.max_iterations - 1:
            self.ctx.iteration += 1
            return self.evaluate()

    def finalize(self):
        """Store final results in AiiDA database."""
        self.out('best_position', List(self.ctx.global_best_position).store())
        self.out('best_value', Float(self.ctx.global_best_value).store())