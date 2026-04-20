.. _static-workflows:

Static Workflows
================

Static workflows are pre-defined AiiDA WorkChains that are registered as entry points
and can be imported by the AiiDA daemon. They work with both ``run()`` and ``submit()``.

Evaluation WorkChains
---------------------

Three static evaluators are shipped with the package:

.. list-table::
   :widths: 40 40 20
   :header-rows: 1

   * - Class
     - Calculator
     - Entry point
   * - ``CrystalLatticeProblem``
     - ``BaseCrystalWorkChain``
     - ``reoptimize.CrystalLatticeProblem``
   * - ``FleurSCFLatticeProblem``
     - ``FleurScfWorkChain``
     - ``reoptimize.FleurSCFLatticeProblem``
   * - ``FleurRelaxLatticeProblem``
     - ``FleurRelaxWorkChain``
     - ``reoptimize.FleurRelaxLatticeProblem``

All three inherit from :class:`~aiida_reoptimize.base.Evaluation.StaticEvalLatticeProblem`
and share the same inputs:

- ``structure`` (``StructureData``): the source crystal structure
- ``targets`` (``List``): list of lattice parameter perturbations to evaluate
- ``calculator_parameters`` (``Dict``): inputs forwarded to the calculator workchain
- ``structure_keyword`` (``List``, optional): path to the structure input in the builder (defaults to ``["structure"]``)

Optimization WorkChains
------------------------

Twenty-one static optimizer workchains are available, combining 7 algorithms with
3 evaluator types:

+------------------+---------------------------+---------------------------+
| Algorithm        | CRYSTAL                   | FLEUR SCF                 |
+==================+===========================+===========================+
| Adam             | ``AdamCrystalOptimizer``  | ``AdamFleurSCFOptimizer`` |
+------------------+---------------------------+---------------------------+
| CDG              | ``CDGCrystalOptimizer``   | ``CDGFleurSCFOptimizer``  |
+------------------+---------------------------+---------------------------+
| RMSprop          | ``RMSpropCrystalOptimizer``| ``RMSpropFleurSCFOptimizer``
+------------------+---------------------------+---------------------------+
| BFGS             | ``BFGSCrystalOptimizer``  | ``BFGSFleurSCFOptimizer`` |
+------------------+---------------------------+---------------------------+
| PyMOO            | ``PyMOOCrystalOptimizer`` | ``PyMOOFleurSCFOptimizer``|
+------------------+---------------------------+---------------------------+
| G3PCX            | ``G3PCXCrystalOptimizer`` | ``G3PCXFleurSCFOptimizer``|
+------------------+---------------------------+---------------------------+
| NRBO             | ``NRBOCrystalOptimizer``  | ``NRBOFleurSCFOptimizer`` |
+------------------+---------------------------+---------------------------+

(The same 7 algorithms are also available for FLEUR relax, e.g.
``AdamFleurRelaxOptimizer``.)

Submitting a Static Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aiida.engine import submit
   from aiida.orm import Dict, Int, StructureData
   from aiida_reoptimize.workflows import BFGSFleurSCFOptimizer

   result = submit(
       BFGSFleurSCFOptimizer,
       parameters=Dict({
           "algorithm_settings": {...},
           "calculator_parameters": {...},
       }),
       structure=my_structure,
       itmax=Int(50),
   )

Adding Custom Static Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a custom evaluator and bind optimizers to it:

.. code-block:: python

   from aiida_reoptimize.base.Evaluation import StaticEvalLatticeProblem
   from aiida_reoptimize.workflows.Optimization._common import (
       AdamOptimizer,
       StaticOptimizerBinding,
   )

   class MyLatticeProblem(StaticEvalLatticeProblem):
       calculator_workchain = MyCalculatorWorkChain

   class BaseMyOptimizer(StaticOptimizerBinding):
       evaluator_workchain = MyLatticeProblem
       extractor_path = ("output_parameters", "energy")

   class AdamMyOptimizer(BaseMyOptimizer, AdamOptimizer):
       pass

Then register the entry points in ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."aiida.workflows"]
   "reoptimize.MyLatticeProblem" = "my_module:MyLatticeProblem"
   "reoptimize.AdamMyOptimizer" = "my_module:AdamMyOptimizer"
