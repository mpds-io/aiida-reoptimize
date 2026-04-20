Dynamic Workflows
=================

Dynamic workflows are created at runtime using
:class:`~aiida_reoptimize.base.OptimizerBuilder.OptimizerBuilder`. They combine an
optimizer, evaluator, and extractor into a single WorkChain class that can be
executed with AiiDA ``run()``.

.. important::
   Dynamic workflows **cannot** be submitted to the AiiDA daemon because the
   generated WorkChain classes are not importable. Use :ref:`static-workflows`
   for daemon-managed or long-running tasks.

OptimizerBuilder
-----------------

:class:`~aiida_reoptimize.base.OptimizerBuilder.OptimizerBuilder` is the main
factory for assembling workflows dynamically.

.. autoclass:: aiida_reoptimize.base.OptimizerBuilder.OptimizerBuilder
   :members:

Factory Methods
~~~~~~~~~~~~~~~

``from_problem``
   Build an optimizer from a simple function-like problem (no structure).

   .. code-block:: python

      from aiida_reoptimize.base import OptimizerBuilder, BasicExtractor
      from aiida_reoptimize.optimizers.convex.GD import AdamOptimizer
      from aiida_reoptimize.problems.problems import Sphere

      builder = OptimizerBuilder.from_problem(
          optimizer_workchain=AdamOptimizer,
          problem_workchain=Sphere,
          extractor=BasicExtractor(lambda outputs: outputs.value.value),
      )
      optimizer = builder.get_optimizer()
      result = run(optimizer, parameters=Dict({"initial_parameters": [1.0, 1.0]}))

``from_ase``
   Build an optimizer from an ASE Atoms structure and a calculator workchain.

   .. code-block:: python

      from ase.build import bulk
      from aiida_reoptimize.base import OptimizerBuilder, BasicExtractor
      from aiida_reoptimize.optimizers.convex.GD import BFGSOptimizer

      builder = OptimizerBuilder.from_ase(
          optimizer_workchain=BFGSOptimizer,
          calculator_workchain=FleurScfWorkChain,
          extractor=BasicExtractor(lambda out: out.output_scf_wc_para.total_energy.value),
          calculator_parameters={...},
          bulk=bulk("SrTiO3", "perovskite", a=3.905),
      )

``from_MPDS``
   Build an optimizer with a structure retrieved from the MPDS database.

   .. code-block:: python

      builder = OptimizerBuilder.from_MPDS(
          optimizer_workchain=BFGSOptimizer,
          calculator_workchain=FleurScfWorkChain,
          extractor=extractor,
          calculator_parameters={...},
          mpds_query="WS2/194",
      )

Evaluator WorkChains
---------------------

Two base evaluator WorkChains are used by the builder:

:class:`~aiida_reoptimize.base.Evaluation.EvalWorkChainProblem`
   For plain parameter-vector problems. Submits a ``problem_workchain`` for each
   target in a batch.

:class:`~aiida_reoptimize.base.Evaluation.EvalWorkChainStructureProblem`
   For structure-based problems. Uses a :class:`~aiida_reoptimize.structure.dynamic_structure.StructureCalculator`
   to generate builders from parameter vectors.

Extractors
----------

:class:`~aiida_reoptimize.base.Extractors.BasicExtractor` walks the outputs of each
finished AiiDA process using a user-supplied function and returns the objective value.
If the process failed or the output is missing, it returns a configurable penalty value
instead.

.. autoclass:: aiida_reoptimize.base.Extractors.BasicExtractor
   :members:
