Structure
=========

Dynamic Structure
------------------

.. autoclass:: aiida_reoptimize.structure.dynamic_structure.DynamicStructure
   :members:

.. autoclass:: aiida_reoptimize.structure.dynamic_structure.StructureCalculator
   :members:

MPDS
-----

.. autofunction:: aiida_reoptimize.structure.MPDS_structure.get_geometry_MPDS

FLEUR Utilities
---------------

.. autoclass:: aiida_reoptimize.structure.fleur_utils.Fleur_setup
   :members:

.. autofunction:: aiida_reoptimize.structure.fleur_utils.convert_xml_to_FleurInpData

Magnetic Moment Utilities
--------------------------

.. autofunction:: aiida_reoptimize.structure.magmoms_utils.convert_to_set
.. autofunction:: aiida_reoptimize.structure.magmoms_utils.check_magmoms_ase
.. autofunction:: aiida_reoptimize.structure.magmoms_utils.numpy_to_python
.. autofunction:: aiida_reoptimize.structure.magmoms_utils.convert_ase_to_spg
.. autofunction:: aiida_reoptimize.structure.magmoms_utils.spg_magnetism_handling
.. autofunction:: aiida_reoptimize.structure.magmoms_utils.spg_get_primitive
.. autofunction:: aiida_reoptimize.structure.magmoms_utils.spg_get_std
.. autofunction:: aiida_reoptimize.structure.magmoms_utils.ase_to_prim
.. autofunction:: aiida_reoptimize.structure.magmoms_utils.ase_to_std
.. autofunction:: aiida_reoptimize.structure.magmoms_utils.numbers_to_symbols
.. autofunction:: aiida_reoptimize.structure.magmoms_utils.ase_to_struct_prim
.. autofunction:: aiida_reoptimize.structure.magmoms_utils.reverse_structure_data
