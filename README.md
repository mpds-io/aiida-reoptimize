# Atomic structure optimization by AiiDA and PyMOO

`aiida-reoptimize` is a framework for running advanced optimization workflows in computational materials science and chemistry, leveraging the AiiDA workflows and the [PyMOO](https://pymoo.org) external optimization library. Both lattice and atomic positions optimizations are supported. Two simulation engines were tested: [CRYSTAL](https://www.crystal.unito.it) and [FLEUR](https://www.flapw.de).


## Features

- **PyMOO integration**: Use state-of-the-art algorithms from the [PyMOO](https://pymoo.org) toolbox.
- **Flexible evaluator system**: Decouple optimization logic from the actual calculation, supporting both simple function optimization and structure-based workflows.
- **Structure optimization**: Easily optimize lattice parameters or atomic positions using the `StructureCalculator` and structure-aware evaluators.
- **Extensible**: allows adding custom optimizers, evaluators, or problem definitions without writing large amounts of code.


## Usage

```sh
git clone https://github.com/mpds-io/aiida-reoptimize.git
cd aiida-reoptimize
pip install .
```

Note dynamic vs. static workflows:

* **Dynamic workflows** (created via `OptimizerBuilder`) allow you to flexibly combine optimizers, evaluators, and extractors at runtime. **However, they can only be used with the AiiDA `run` (not `submit`) method, because they are not importable by the AiiDA daemon.**
This means they are suitable for interactive or short-running tasks, but not for long-running or daemon-managed workflows.

* **Static workflows** (in [workflows](https://github.com/mpds-io/aiida-reoptimize/tree/master/aiida_reoptimize/workflows)) are pre-defined, importable workflows registered as AiiDA entry points.
**They can be used with both `run` and `submit` methods and are suitable for production, daemon-managed, or long-running tasks.**
Currently available static workflows require to specify the crystal structure as an input parameter (see below).

### Static workchains

The package ships importable static AiiDA workchains under [aiida_reoptimize/workflows](aiida_reoptimize/workflows). They are suitable for `submit` and for daemon execution.

- Evaluation workchains:
  - `aiida_reoptimize.CrystalLatticeProblem`: evaluate lattice distortions with `BaseCrystalWorkChain`
  - `aiida_reoptimize.FleurSCFLatticeProblem`: evaluate lattice distortions with `FleurScfWorkChain`
  - `aiida_reoptimize.FleurRelaxLatticeProblem`: evaluate lattice distortions with `FleurRelaxWorkChain`
- Optimization workchains:
  - CRYSTAL: `AdamCrystalOptimizer`, `CDGCrystalOptimizer`, `RMSpropCrystalOptimizer`, `BFGSCrystalOptimizer`, `PyMOOCrystalOptimizer`, `G3PCXCrystalOptimizer`, `NRBOCrystalOptimizer`
  - FLEUR SCF: `AdamFleurSCFOptimizer`, `CDGFleurSCFOptimizer`, `RMSpropFleurSCFOptimizer`, `BFGSFleurSCFOptimizer`, `PyMOOFleurSCFOptimizer`, `G3PCXFleurSCFOptimizer`, `NRBOFleurSCFOptimizer`
  - FLEUR relax: `AdamFleurRelaxOptimizer`, `CDGFleurRelaxOptimizer`, `RMSpropFleurRelaxOptimizer`, `BFGSFleurRelaxOptimizer`, `PyMOOFleurRelaxOptimizer`, `G3PCXFleurRelaxOptimizer`, `NRBOFleurRelaxOptimizer`

Static lattice evaluators share the same core inputs:

- `structure`: source `StructureData`
- `targets`: list of lattice perturbations to evaluate
- `calculator_parameters`: inputs forwarded to the underlying calculator workchain
- `structure_keyword`: optional path to the structure input in the target builder, defaults to `['structure']`

### How To Add A Custom Static Workflow

Minimal pattern for adding your own static evaluator and optimizer while keeping compatibility with `submit` and AiiDA entry points.

```python
from aiida_reoptimize.base.Evaluation import StaticEvalLatticeProblem
from aiida_reoptimize.workflows.Optimization._common import (
  AdamOptimizer,
  StaticOptimizerBinding,
)


class MyLatticeProblem(StaticEvalLatticeProblem):
  # Any AiiDA workchain that accepts structure + calculator parameters.
  calculator_workchain = MyCalculatorWorkChain


class BaseMyOptimizer(StaticOptimizerBinding):
  evaluator_workchain = MyLatticeProblem
  # Path inside node.outputs used as objective value.
  extractor_path = ("output_parameters", "energy")


class AdamMyOptimizer(BaseMyOptimizer, AdamOptimizer):
  pass
```

Then register your classes in `pyproject.toml` under `project.entry-points."aiida.workflows"`.

```toml
[project.entry-points."aiida.workflows"]
"aiida_reoptimize.MyLatticeProblem" = "your_module:MyLatticeProblem"
"aiida_reoptimize.AdamMyOptimizer" = "your_module:AdamMyOptimizer"
```


## Technical details

- **PyMOO**: Its objects are kept as local variables in the WorkChain (not in the AiiDA context) to avoid serialization issues.
- **Evaluator WorkChains**: Optimization is performed by submitting batches of calculations via a dedicated evaluator WorkChain. The evaluator:
  - Accepts a problem WorkChain (e.g., a function or structure calculation);
  - Receives a list of parameter sets to evaluate;
  - Uses an extractor function to obtain the relevant result from each calculation;
  - Handles penalties for failed calculations.
- **Structure Optimization**: For crystal structure-based problems, the `StructureCalculator` class generates new structures (with updated lattice parameters) and returns a builder for the corresponding calculation WorkChain.
- **Two Evaluator Types**:
  - **Parameter optimizers**: Directly optimize numerical parameters.
  - **Structure optimizers**: Modify and optimize crystal structures.

Currently, two types of algorithms for structure optimization are implemented.

### Gradient-based optimizers

These optimizers are implemented as AiiDA WorkChains:

- **BFGS**
- **Adam**
- **RMSProp**

#### Their common input parameters:

- `itmax` (`Int`): Maximal number of iterations (default: `100`)
- `parameters` (`Dict`): Dictionary containing algorithm-specific settings (in `algorithm_settings`, `Dict`) and additional parameters required for optimization
- `get_best` (`Bool`, optional): Whether to return the best result node identifier (default: `True`)
- `structure` (`StructureData`, optional): Chemical structure for the optimization (required for so-called static workflows)

### Convex algorithms

All convex optimizers accept the following parameters (passed as a `Dict` under the `parameters` input):

| Parameter             | Type    | Description                                              |
|-----------------------|---------|----------------------------------------------------------|
| `algorithm_settings`  | Dict    | Algorithm-specific settings (see below)                  |
| `initial_parameters`  | List    | Initial guess for the parameters to optimize             |

**Gradient descent-based algorithms** (`algorithm_settings`):

- `tolerance` (`float`): Convergence threshold for gradient norm (default: `1e-3`)
- `epsilon` (`float`): Small value to avoid division by zero (default: `1e-7`)
- `delta` (`float`): Step size for numerical gradient (default: `5e-4`)

**RMSProp-specific settings** (`algorithm_settings`):

- `learning_rate` (`float`): Step size for parameter updates (default: `1e-3`)
- `rho` (`float`): Decay rate for moving average (default: `0.9`)

**Adam-specific settings** (`algorithm_settings`):

- `learning_rate` (`float`): Step size for parameter updates (default: `5e-2`)
- `beta1` (`float`): Exponential decay rate for the first moment estimates (default: `0.5`)
- `beta2` (`float`): Exponential decay rate for the second moment estimates (default: `0.999`)

**BFGS-specific settings** (`algorithm_settings`):

- `alpha` (`float`): Initial step size for the line search (default: `1.0`)
- `beta` (`float`): Step size reduction factor (default: `0.5`)
- `sigma` (`float`): Armijo/sufficient decrease parameter (default: `1e-4`)
- `linesearch_max_iter` (`int`): Maximum allowed steps in line search (default: `20`)

---

### Algorithms provided by PyMOO library

The `PyMOO_Optimizer` class requires the following parameters as input:

- `itmax` (`Int`): Maximal number of iterations
- `algorithm_name` (`Str`): Name of the PyMOO algorithm to use (see below)
- `parameters` (`Dict`): Contains `algorithm_settings` and other required parameters

**Inside `parameters` (`Dict`):**

- `algorithm_settings` (`Dict`): Algorithm-specific settings (see below)
- `bounds` (`List`): List of [min, max] for each variable
- `dimensions` (`Int`): Number of variables to optimize

**Supported PyMOO algorithms (`algorithm_name`):**

- `DE` (Differential Evolution)
- `ES` (Evolution Strategy)
- `GA` (Genetic Algorithm)
- `G3PCX` (Generalized Generation Gap with Parent-Centric Crossover)
- `NRBO` (Neighborhood-Ranked Bayesian Optimization)
- `PSO` (Particle Swarm Optimization)

**Algorithm-specific settings** (inside `algorithm_settings`):

- **DE**: `pop_size`, `n_offsprings`, `sampling`, `variant`
- **ES**: `pop_size`, `n_offsprings`, `rule`, `phi`, `gamma`, `sampling`
- **GA**: `pop_size`, `termination`, `sampling`, `selection`, `crossover`, `mutation`, `eliminate_duplicates`, `n_offsprings`
- **G3PCX**: `pop_size`, `sampling`, `n_offsprings`, `n_parents`, `family_size`, `repair`
- **PSO**: `pop_size`, `sampling`, `w`, `c1`, `c2`, `adaptive`, `initial_velocity`, `max_velocity_rate`, `pertube_best`

For details see the [PyMOO docs](https://pymoo.org/algorithms/list.html).

**Note:**

When specifying operators such as `sampling`, `selection`, `crossover`, `mutation`, or `repair` in your `algorithm_settings`,
**you should use the name of the operator as a string** (e.g., `"SBX"`, `"FRS"`, `"TOS"`),
**not** an instance of the operator class.

**Example:**

```python
parameters = Dict({
    "algorithm_name": "GA",
    "algorithm_settings": {
        "pop_size": 50,
        "sampling": "LHS",
        "crossover": "SBX",
        "mutation": "PM",
        "selection": "TOS"
    },
    "bounds": [[0, 1], [0, 1]],
    "dimensions": 2
```


## Examples

For detailed examples of structure optimization workflows, refer to the `examples` directory.


## References

- [pymoo: Multi-objective Optimization in Python](https://pymoo.org)
- [AiiDA: Automated Interactive Infrastructure and Database for Computational Science](https://www.aiida.net)
- [Another work on structure optimization in AiiDA by Dominik Gresch](https://github.com/greschd/aiida-optimize)


## License

MIT

&copy; 2025-2026 Materials Platform for Data Science OÜ
