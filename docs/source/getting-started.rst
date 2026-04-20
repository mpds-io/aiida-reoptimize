Getting Started
===============

Installation
------------

Install from source::

   git clone https://github.com/mpds-io/aiida-reoptimize.git
   cd aiida-reoptimize
   pip install .

Documentation dependencies can be installed with::

   pip install ".[docs]"

Quick Start
-----------

The package offers two workflow paradigms:

**Dynamic workflows** are assembled at runtime via :class:`~aiida_reoptimize.base.OptimizerBuilder.OptimizerBuilder`.
They can only be used with AiiDA ``run()`` (not ``submit()``), making them suitable for
interactive or short-running tasks.

**Static workflows** are pre-defined, importable workchains registered as AiiDA entry points.
They work with both ``run()`` and ``submit()`` and are suitable for daemon-managed production use.

Dynamic workflow example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aiida.engine import run
   from aiida_reoptimize.base import OptimizerBuilder, BasicExtractor
   from aiida_reoptimize.optimizers.convex.GD import AdamOptimizer
   from aiida_reoptimize.problems.problems import Sphere

   extractor = BasicExtractor(lambda outputs: outputs.value.value)
   builder = OptimizerBuilder.from_problem(
       optimizer_workchain=AdamOptimizer,
       problem_workchain=Sphere,
       extractor=extractor,
   )
   optimizer = builder.get_optimizer()
   result = run(optimizer, parameters=Dict({"initial_parameters": [1.0, 1.0]}))

Static workflow example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aiida.engine import submit
   from aiida.orm import Dict, Int, Str, StructureData
   from aiida_reoptimize.workflows import AdamFleurSCFOptimizer

   result = submit(
       AdamFleurSCFOptimizer,
       parameters=Dict({...}),
       structure=StructureData(...),
       itmax=Int(50),
   )
