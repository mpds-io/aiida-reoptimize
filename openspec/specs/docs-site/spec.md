### Requirement: Sphinx documentation site
The project SHALL provide a Sphinx-based documentation site in a `docs/` directory at the project root with a `source/conf.py` configuration, enabling HTML generation via `sphinx-build` or `make html`.

#### Scenario: Building the documentation
- **WHEN** a user runs `sphinx-build docs/source docs/build` or `make html` in the `docs/` directory
- **THEN** an HTML documentation site is generated without errors in the build output directory

### Requirement: API reference from docstrings
The documentation site SHALL include an auto-generated API reference section that documents all public classes, functions, and methods in `aiida_reoptimize/` using Sphinx autodoc, rendering docstrings and type annotations.

#### Scenario: API reference covers base module
- **WHEN** the API reference page for `aiida_reoptimize.base` is rendered
- **THEN** it includes documentation for `EvalWorkChainProblem`, `EvalWorkChainStructureProblem`, `StaticEvalLatticeProblem`, `OptimizerBuilder`, `BasicExtractor`, and `find_nodes`

#### Scenario: API reference covers optimizers module
- **WHEN** the API reference page for `aiida_reoptimize.optimizers` is rendered
- **THEN** it includes documentation for `_OptimizerBase`, `AdamOptimizer`, `RMSpropOptimizer`, `ConjugateGradientOptimizer`, `BFGSOptimizer`, `PyMOO_Optimizer`, `AlgorithmBuilder`, and key utility functions

#### Scenario: API reference covers structure module
- **WHEN** the API reference page for `aiida_reoptimize.structure` is rendered
- **THEN** it includes documentation for `DynamicStructure`, `StructureCalculator`, `get_geometry_MPDS`, `Fleur_setup`, and magnetic moment utility functions

### Requirement: Getting started guide
The documentation site SHALL include a getting-started page that covers installation instructions and a minimal quick-start example.

#### Scenario: New user follows getting-started guide
- **WHEN** a new user reads the getting-started page
- **THEN** they can install the package and run a minimal optimization example using the instructions provided

### Requirement: Dynamic workflows guide
The documentation site SHALL include a usage guide for dynamic workflows (`OptimizerBuilder`), explaining the `from_problem()`, `from_ase()`, and `from_MPDS()` factory methods and their caveats (cannot be submitted to daemon).

#### Scenario: User learns dynamic workflow setup
- **WHEN** a user reads the dynamic workflows guide
- **THEN** they understand how to use `OptimizerBuilder` to assemble an optimizer at runtime and run it with AiiDA `run()`

### Requirement: Static workflows guide
The documentation site SHALL include a usage guide for static workflows, listing all registered entry points and explaining how to submit them to the AiiDA daemon.

#### Scenario: User learns static workflow submission
- **WHEN** a user reads the static workflows guide
- **THEN** they can identify the correct entry point name and submit a static optimizer workchain via `submit()`

### Requirement: Optimizers reference
The documentation site SHALL include a reference page for all optimizer algorithms, documenting their parameters, convergence behavior, and applicable problem types.

#### Scenario: User selects an optimizer
- **WHEN** a user reads the optimizers reference
- **THEN** they can compare available algorithms and understand which parameters each algorithm accepts

### Requirement: Structure utilities guide
The documentation site SHALL include a guide for structure manipulation utilities (`DynamicStructure`, `StructureCalculator`, MPDS retrieval, FLEUR setup, magnetic moment handling).

#### Scenario: User learns structure utilities
- **WHEN** a user reads the structure utilities guide
- **THEN** they understand how to retrieve structures from MPDS, create a `StructureCalculator`, and handle magnetic moments

### Requirement: Examples catalog
The documentation site SHALL include an examples index page that catalogs all example scripts in the `examples/` directory, grouped by category, with brief descriptions.

#### Scenario: User browses available examples
- **WHEN** a user reads the examples index
- **THEN** they see a categorized list of all example scripts with descriptions of what each demonstrates

### Requirement: Docstrings for public API
All public classes and functions in `aiida_reoptimize/` SHALL have Google-style docstrings with a summary line, parameter descriptions, and return value descriptions where applicable.

#### Scenario: Missing docstring detected
- **WHEN** a class or function with a public name in `aiida_reoptimize/` lacks a docstring
- **THEN** the Sphinx build emits a warning for that missing docstring
