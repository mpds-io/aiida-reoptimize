Optimizers
==========

The package provides two families of optimization algorithms:

.. toctree::
   :maxdepth: 2

Gradient-based Optimizers
-------------------------

All gradient-based optimizers inherit from :class:`~aiida_reoptimize.optimizers.OptimizerBase._OptimizerBase`
and compute numerical gradients via finite differences.

Common Inputs
~~~~~~~~~~~~~

All gradient-based optimizers share these inputs:

- ``parameters`` (``Dict``): optimization parameters, including ``initial_parameters`` and ``algorithm_settings``
- ``itmax`` (``Int``, default 100): maximum number of iterations
- ``get_best`` (``Bool``, default True): whether to return the best result node PK
- ``structure`` (``StructureData``, optional): structure for lattice optimization

Common Outputs
~~~~~~~~~~~~~~

- ``optimized_parameters`` (``List``): the optimized parameter vector
- ``final_value`` (``Float``): the final objective function value
- ``history`` (``List``): iteration history
- ``result_node_pk`` (``Int``, optional): PK of the best AiiDA node

Convergence and Robustness
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All gradient-based optimizers include:

- **Step clamping**: limits step size to ``max_step`` (default 0.1) to prevent instability
- **Rollback on worse objective**: reverts parameters if the objective increases
- **Learning rate decay**: reduces the step rate when stuck
- **Random jump escape**: adds random perturbation when stuck for ``allowed_stuck`` consecutive iterations

Adam
~~~~

Adaptive moment estimation optimizer.

Algorithm settings (inside ``algorithm_settings``):

+--------------------+--------+------------------------------------------+
| Parameter          | Type   | Default                                  |
+====================+========+==========================================+
| ``learning_rate``  | float  | 5e-2                                     |
+--------------------+--------+------------------------------------------+
| ``beta1``          | float  | 0.5                                      |
+--------------------+--------+------------------------------------------+
| ``beta2``          | float  | 0.999                                    |
+--------------------+--------+------------------------------------------+
| ``tolerance``      | float  | 1e-3                                     |
+--------------------+--------+------------------------------------------+
| ``epsilon``        | float  | 1e-7                                     |
+--------------------+--------+------------------------------------------+
| ``delta``          | float  | 1e-6                                     |
+--------------------+--------+------------------------------------------+

.. autoclass:: aiida_reoptimize.optimizers.convex.GD.AdamOptimizer
   :members:

RMSprop
~~~~~~~

Root mean square propagation optimizer.

Algorithm settings:

+--------------------+--------+------------------------------------------+
| Parameter          | Type   | Default                                  |
+====================+========+==========================================+
| ``learning_rate``  | float  | 1e-3                                     |
+--------------------+--------+------------------------------------------+
| ``rho``            | float  | 0.9                                      |
+--------------------+--------+------------------------------------------+
| ``tolerance``      | float  | 1e-3                                     |
+--------------------+--------+------------------------------------------+
| ``epsilon``        | float  | 1e-7                                     |
+--------------------+--------+------------------------------------------+
| ``delta``          | float  | 1e-6                                     |
+--------------------+--------+------------------------------------------+

.. autoclass:: aiida_reoptimize.optimizers.convex.GD.RMSpropOptimizer
   :members:

Conjugate Gradient (Polak-Ribiere)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conjugate gradient optimizer with dynamic learning rate and periodic restarts.

Algorithm settings:

+----------------------+--------+------------------------------------------+
| Parameter            | Type   | Default                                  |
+======================+========+==========================================+
| ``learning_rate``    | float  | 1e-2                                     |
+----------------------+--------+------------------------------------------+
| ``lr_min``           | float  | 1e-8                                     |
+----------------------+--------+------------------------------------------+
| ``lr_max``           | float  | 1.0                                      |
+----------------------+--------+------------------------------------------+
| ``lr_increase``      | float  | 1.1                                      |
+----------------------+--------+------------------------------------------+
| ``lr_decrease``      | float  | 0.5                                      |
+----------------------+--------+------------------------------------------+
| ``restart_interval`` | int    | 10                                       |
+----------------------+--------+------------------------------------------+
| ``tolerance``        | float  | 1e-3                                     |
+----------------------+--------+------------------------------------------+
| ``epsilon``          | float  | 1e-7                                     |
+----------------------+--------+------------------------------------------+
| ``delta``            | float  | 1e-6                                     |
+----------------------+--------+------------------------------------------+

.. autoclass:: aiida_reoptimize.optimizers.convex.GD.ConjugateGradientOptimizer
   :members:

BFGS
~~~~~

BFGS quasi-Newton optimizer with backtracking line search (Armijo condition).

Algorithm settings:

+------------------------+--------+------------------------------------------+
| Parameter              | Type   | Default                                  |
+========================+========+==========================================+
| ``alpha``              | float  | 1.0                                      |
+------------------------+--------+------------------------------------------+
| ``beta``               | float  | 0.5                                      |
+------------------------+--------+------------------------------------------+
| ``sigma``              | float  | 1e-4                                     |
+------------------------+--------+------------------------------------------+
| ``linesearch_max_iter``| int    | 20                                       |
+------------------------+--------+------------------------------------------+
| ``tolerance``          | float  | 1e-3                                     |
+------------------------+--------+------------------------------------------+
| ``epsilon``            | float  | 1e-7                                     |
+------------------------+--------+------------------------------------------+
| ``delta``              | float  | 1e-6                                     |
+------------------------+--------+------------------------------------------+

.. autoclass:: aiida_reoptimize.optimizers.convex.QN.BFGSOptimizer
   :members:

PyMOO Optimizers
-----------------

:class:`~aiida_reoptimize.optimizers.PyMOO.PyMOO.PyMOO_Optimizer` wraps algorithms
from the PyMOO library in an AiiDA WorkChain. It uses an ask-evaluate-tell loop
where each iteration submits a batch of evaluations.

Inputs
~~~~~~~

- ``algorithm_name`` (``Str``): name of the PyMOO algorithm
- ``parameters`` (``Dict``): contains ``bounds``, ``algorithm_settings``, and optional ``tol``
- ``itmax`` (``Int``): maximum number of iterations
- ``itmin`` (``Int``, default 10): minimum iterations before early stopping via ``tol``
- ``structure`` (``StructureData``, optional): structure for lattice optimization
- ``get_best`` (``Bool``, default True): whether to return the best result PK

Early Stopping
~~~~~~~~~~~~~~

If ``tol`` is specified, the optimizer stops when the spread of the best objective
values over 3 consecutive iterations falls below ``tol`` (after at least ``itmin``
iterations).

Bounds
~~~~~~

``bounds`` can be:

- A **scalar multiplier** ``b``: each parameter ``p`` gets bounds ``(-b*p, b*p)``
- A **list of [low, high] pairs**: explicit bounds for each variable

The number of optimized variables is inferred automatically from the Bravais
lattice (when a structure is provided), from ``initial_parameters``, or from the
length of the bounds list.

Supported Algorithms
~~~~~~~~~~~~~~~~~~~~

+---------+----------------------------------------------+
| Name    | Description                                  |
+=========+==============================================+
| ``DE``  | Differential Evolution                       |
+---------+----------------------------------------------+
| ``ES``  | Evolution Strategy                           |
+---------+----------------------------------------------+
| ``GA``  | Genetic Algorithm                            |
+---------+----------------------------------------------+
| ``G3PCX``| Generalized Generation Gap with Parent-Centric Crossover |
+---------+----------------------------------------------+
| ``NRBO``| Neighborhood-Ranked Bayesian Optimization    |
+---------+----------------------------------------------+
| ``PSO`` | Particle Swarm Optimization                  |
+---------+----------------------------------------------+

Algorithm Settings
~~~~~~~~~~~~~~~~~~~

Each algorithm accepts specific keyword arguments inside ``algorithm_settings``
(see :class:`~aiida_reoptimize.optimizers.PyMOO.Builder.AlgorithmBuilder`).
Use string names for operators (e.g., ``"SBX"`` for crossover, ``"FRS"`` for
sampling, ``"TOS"`` for selection).

.. autoclass:: aiida_reoptimize.optimizers.PyMOO.PyMOO.PyMOO_Optimizer
   :members:

AlgorithmBuilder
~~~~~~~~~~~~~~~~~

.. autoclass:: aiida_reoptimize.optimizers.PyMOO.Builder.AlgorithmBuilder
   :members:

Parameter Normalization
-------------------------

.. autofunction:: aiida_reoptimize.optimizers.parameter_utils.prepare_optimization_parameters
