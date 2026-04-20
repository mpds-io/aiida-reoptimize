Structure Utilities
===================

The ``aiida_reoptimize.structure`` module provides tools for crystal structure
manipulation, retrieval from databases, and conversion between ASE and AiiDA
representations.

DynamicStructure
-----------------

:class:`~aiida_reoptimize.structure.dynamic_structure.DynamicStructure` generates
new structures from a given ASE Atoms object by replacing the unit cell with a
new Bravais lattice constructed from parameter vector ``x``.

.. autoclass:: aiida_reoptimize.structure.dynamic_structure.DynamicStructure
   :members:

StructureCalculator
--------------------

:class:`~aiida_reoptimize.structure.dynamic_structure.StructureCalculator` combines
a :class:`~aiida_reoptimize.structure.dynamic_structure.DynamicStructure` with a
calculator WorkChain. It produces a ready-to-submit process builder for each
parameter vector, inserting the modified structure at a configurable nested path.

.. autoclass:: aiida_reoptimize.structure.dynamic_structure.StructureCalculator
   :members:

MPDS Structure Retrieval
-------------------------

.. autofunction:: aiida_reoptimize.structure.MPDS_structure.get_geometry_MPDS

FLEUR Utilities
----------------

.. autoclass:: aiida_reoptimize.structure.fleur_utils.Fleur_setup
   :members:

.. autofunction:: aiida_reoptimize.structure.fleur_utils.convert_xml_to_FleurInpData

Magnetic Moment Utilities
-------------------------

These functions handle conversion between ASE Atoms and standardized/primitive
cells while preserving magnetic moments, using spglib for symmetry operations.

.. autofunction:: aiida_reoptimize.structure.magmoms_utils.ase_to_prim

.. autofunction:: aiida_reoptimize.structure.magmoms_utils.ase_to_std

.. autofunction:: aiida_reoptimize.structure.magmoms_utils.ase_to_struct_prim

.. autofunction:: aiida_reoptimize.structure.magmoms_utils.reverse_structure_data

.. autofunction:: aiida_reoptimize.structure.magmoms_utils.convert_to_set

.. autofunction:: aiida_reoptimize.structure.magmoms_utils.check_magmoms_ase

.. autofunction:: aiida_reoptimize.structure.magmoms_utils.numpy_to_python

.. autofunction:: aiida_reoptimize.structure.magmoms_utils.convert_ase_to_spg

.. autofunction:: aiida_reoptimize.structure.magmoms_utils.spg_magnetism_handling

.. autofunction:: aiida_reoptimize.structure.magmoms_utils.spg_get_primitive

.. autofunction:: aiida_reoptimize.structure.magmoms_utils.spg_get_std

.. autofunction:: aiida_reoptimize.structure.magmoms_utils.numbers_to_symbols
