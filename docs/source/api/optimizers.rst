Optimizers
==========

Optimizer Base
---------------

.. autoclass:: aiida_reoptimize.optimizers.OptimizerBase._OptimizerBase
   :members:

Gradient-based
--------------

.. autoclass:: aiida_reoptimize.optimizers.convex.base._GDBase
   :members:

.. autoclass:: aiida_reoptimize.optimizers.convex.GD.AdamOptimizer
   :members:

.. autoclass:: aiida_reoptimize.optimizers.convex.GD.RMSpropOptimizer
   :members:

.. autoclass:: aiida_reoptimize.optimizers.convex.GD.ConjugateGradientOptimizer
   :members:

.. autoclass:: aiida_reoptimize.optimizers.convex.QN.BFGSOptimizer
   :members:

PyMOO
------

.. autoclass:: aiida_reoptimize.optimizers.PyMOO.PyMOO._PyMOO_Base
   :members:

.. autoclass:: aiida_reoptimize.optimizers.PyMOO.PyMOO.PyMOO_Optimizer
   :members:

.. autoclass:: aiida_reoptimize.optimizers.PyMOO.Builder.AlgorithmBuilder
   :members:

Utilities
----------

.. autofunction:: aiida_reoptimize.optimizers.parameter_utils.prepare_optimization_parameters

.. autofunction:: aiida_reoptimize.optimizers.result_utils.ensure_population_has_valid_results
