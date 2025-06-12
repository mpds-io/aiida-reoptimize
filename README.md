# aiida-reoptimize

**Atomic structure and parameter optimization powered by AiiDA and PyMOO**

---

## Overview

`aiida-reoptimize` is a flexible framework for running advanced optimization workflows in computational materials science and chemistry, leveraging the AiiDA workflow engine and the PyMOO optimization library. It supports both parameter and atomic structure optimization, and is designed for easy integration with external simulation codes and custom workflows.

---

## Features

- **PyMOO integration**: Use state-of-the-art algorithms from the PyMOO library.
- **Flexible evaluator system**: Decouple optimization logic from the actual calculation, supporting both simple function optimization and structure-based workflows.
- **Structure optimization**: Easily optimize lattice parameters or atomic positions using the `StructureCalculator` and structure-aware evaluators.
- **Extensible**: allows you to add custom optimizers, evaluators, or problem definitions without writing large amounts of code.

---

## Technical Details

- **PyMOO**: The package uses PyMOO library for optimization. These objects are kept as local variables in the WorkChain (not in the AiiDA context) to avoid serialization issues.
- **Evaluator WorkChains**: Optimization is performed by submitting batches of calculations via a dedicated evaluator WorkChain. The evaluator:
  - Accepts a problem WorkChain (e.g., a function or structure calculation);
  - Receives a list of parameter sets to evaluate;
  - Uses an extractor function to obtain the relevant result from each calculation;
  - Handles penalties for failed calculations.
- **Structure Optimization**: For crystal structure-based problems, the `StructureCalculator` class generates new structures (with updated lattice parameters) and returns a builder for the corresponding calculation WorkChain.
- **Two Evaluator Types**:
  - **Parameter optimizers**: Directly optimize numerical parameters.
  - **Structure optimizers**: Modify and optimize crystal structures.

---
## Algorithms

Currently, two types of algorithms are implemented.

### Gradient-based optimizers

These optimizers are implemented as AiiDA WorkChains:

- **BFGSOptimizer**
- **AdamOptimizer**
- **RMSPropOptimizer**

## Input parameters of the optimizers

All implemented algorithms accept a common parameters:
- `itmax` (Int): maximal number of iterations (default: `100`)
- `parameters` (Dict): Dictionary that contains algorithms specific settings (in `algorithm_settings`, `Dict`) and additional parameters required for optimization

### Convex algorithms

All convex optimizers accept the following parameters (passed as a `Dict` under the `parameters` input):

| Parameter             | Type    | Description                                              |
|-----------------------|---------|----------------------------------------------------------|
| `algorithm_settings`  | Dict    | Algorithm-specific settings (see below)                  |
| `initial_parameters`  | List    | Initial guess for the parameters to optimize             |

**Gradient descent based algorithms** (inside `algorithm_settings`):
- `tolerance` (float): Convergence threshold for gradient norm (default: `1e-3`)
- `epsilon` (float): Small value to avoid division by zero (default: `1e-10`)
- `delta` (float): Step size for numerical gradient (default: `1e-6`)

**RMSProp-specific settings** (inside `algorithm_settings`):
- `learning_rate` (float): Step size for parameter updates (default: `1e-3`)
- `rho` (float): Decay rate for moving average (default: `0.9`)

**Adam-specific settings** (inside `algorithm_settings`):
- `learning_rate` (float): Step size for parameter updates (default: `1e-3`)
- `beta1`, `beta2` (float): Exponential decay rates for moment estimates (default: `0.9`, `0.999`)


**BFGS-specific settings** (inside `algorithm_settings`):
- `alpha` (float): Initial step size for the line search procedure, which influences the starting magnitude of parameter updates. A larger value may speed up convergence but risks overshooting, while a smaller value ensures stability at the cost of slower progress (default: `1.0`).
- `beta` (float): Step size reduction factor (default: `0.5`)
- `sigma` (float): Armijo/sufficient decrease parameter. This controls how much decrease in the objective function is considered "sufficient." (default: `1e-4`)
- `linesearch_max_iter` (int): Maximum allowed steps in line search procedure (default: `20`)

### Algorithms provided by PyMOO library

`PyMOO_Optimizer` class requires the following parameters to be specified as an input:
- `parameters` (Dict): Contain `algorithm_settings` dict and other parameters (see below);
- `itmax` (Int): Maximal number of iterations;
- `algorithm_name` (Str): Name of algorithm required for optimization. `aiida-reoptimize` currently supports:
  - `DE` (Differential Evolution): A population-based optimization algorithm suitable for non-linear and non-differentiable functions. [Learn more](https://pymoo.org/algorithms/soo/de.html)
  - `ES` (Evolution Strategy): A stochastic optimization method inspired by natural evolution. [Learn more](https://pymoo.org/algorithms/soo/es.html)
  - `GA` (Genetic Algorithm): A heuristic search algorithm based on the principles of genetics and natural selection. [Learn more](https://pymoo.org/algorithms/soo/ga.html)
  - `G3PCX` (Generalized Generation Gap with Parent-Centric Crossover): An advanced evolutionary algorithm for complex optimization problems. [Learn more](https://pymoo.org/algorithms/soo/g3pcx.html)
  - `PSO` (Particle Swarm Optimization): A computational method that optimizes a problem by iteratively improving candidate solutions based on the movement of particles. [Learn more](https://pymoo.org/algorithms/soo/pso.html)

Common parameters for optimization algorithms implemented via PyMOO library (inside the `parameters` dict):
- `bounds` (List): List of lists, where each inner list contains the minimum and maximum allowed values for a single optimization variable;
- `dimensions` (Int): Specifies the number of variables or dimensions in the optimization problem, defining the size of the search space.

When using optimizers implemented via the PyMOO library, you can customize the algorithm behavior by providing algorithm-specific keywords in the `algorithm_settings` dictionary.
The following keywords are supported for each algorithm:

- **DE**: `pop_size`, `n_offsprings`, `sampling`, `variant`
- **ES**: `pop_size`, `n_offsprings`, `rule`, `phi`, `gamma`, `sampling`
- **GA**: `pop_size`, `termination`, `sampling`, `selection`, `crossover`, `mutation`, `eliminate_duplicates`, `n_offsprings`
- **G3PCX**: `pop_size`, `sampling`, `n_offsprings`, `n_parents`, `family_size`, `repair`
- **PSO**: `pop_size`, `sampling`, `w`, `c1`, `c2`, `adaptive`, `initial_velocity`, `max_velocity_rate`, `pertube_best`

For details on the meaning and possible values of each keyword, see the [PyMOO documentation](https://pymoo.org/algorithms/list.html).

**Note:**  
When specifying operators such as `sampling`, `selection`, `crossover`, `mutation`, or `repair` in your `algorithm_settings`,  
**you should use the name of the operator as a string** (e.g., `"SBX"`, `"FRS"`, `"TOS"`),  
**not** an instance of the operator class.

For example:
```python
parameters = Dict({
    "algorithm_name": "GA",
    "algorithm_settings": {
        "pop_size": 50,
        "sampling": "LHS",         # Use "LHS" (Latin Hypercube Sampling) instead of LHS()
        "crossover": "SBX",        # Use "SBX" instead of SBX()
        "mutation": "PM",          # Use "PM" instead of PolynomialMutation()
        "selection": "TOS"         # Use "TOS" instead of TournamentSelection()
    },
    ...
})
```

---

## Installation

```sh
git clone https://github.com/mpds-io/aiida-reoptimize.git
cd aiida-reoptimize
pip install .
```

---

## Usage

### 1. Define Your Problem

For a simple function optimization:
```python
from aiida_reoptimize.problems.problems import Sphere
```

### 2. Build an Optimizer Pipeline

```python
from aiida_reoptimize.base.OptimizerBuilder import OptimizerBuilder
from aiida_reoptimize.optimizers.convex.QN import BFGSOptimizer

builder = OptimizerBuilder.from_problem(
    optimizer_workchain=BFGSOptimizer,
    problem_workchain=Sphere,
    extractor=lambda x: x["value"],
)
optimizer = builder.get_optimizer()
```

### 3. Run the Optimization

```python
from aiida.engine import run
from aiida.orm import Dict, Int, List

optimizer_parameters = {
    "itmax": Int(20),
    "parameters": Dict({
        "algorithm_settings": {"tolerance": 1e-8},
        "initial_parameters": List([0.1, -0.3, 0.7]),
    })
}
results = run(optimizer, **optimizer_parameters)
```

---

## Example: Structure Optimization

For detailed examples of structure optimization workflows, refer to the [examples directory](https://github.com/mpds-io/aiida-reoptimize/tree/master/examples). It contains sample scripts and configurations to help you get started with optimizing atomic structures and parameters using `aiida-reoptimize`.

---

## References

- [pymoo: Multi-objective Optimization in Python](https://pymoo.org)
- [AiiDA: Automated Interactive Infrastructure and Database for Computational Science](https://www.aiida.net)
- [Similar work by Dominik Gresch](https://github.com/greschd/aiida-optimize)

---

## License

MIT

&copy; 2025 Tilde Materials Informatics and Materials Platform for Data Science LLC
