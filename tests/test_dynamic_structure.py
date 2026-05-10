import unittest

import numpy as np
from ase import Atoms
from ase.lattice import BCT, MCL, ORC

from aiida_reoptimize.structure.dynamic_structure import DynamicStructure, ParameterVectorMismatchError


class TestDynamicStructure(unittest.TestCase):
    @staticmethod
    def atoms_for_lattice(lattice):
        return Atoms("Mg", scaled_positions=[[0.0, 0.0, 0.0]], cell=lattice.tocell(), pbc=True)

    def assert_cell_allclose(self, atoms, expected):
        np.testing.assert_allclose(atoms.cell.array, expected, atol=1e-12)

    def test_orc_boundary_equal_axes_generates_structure(self):
        dynamic_structure = DynamicStructure(self.atoms_for_lattice(ORC(1.0, 2.0, 3.0)))

        generated = dynamic_structure([2.0, 2.0, 3.0])

        self.assert_cell_allclose(generated, np.diag([2.0, 2.0, 3.0]))

    def test_orc_unsorted_axes_are_not_reordered_by_standardization(self):
        dynamic_structure = DynamicStructure(self.atoms_for_lattice(ORC(1.0, 2.0, 3.0)))

        generated = dynamic_structure([2.5, 1.5, 3.0])

        self.assert_cell_allclose(generated, np.diag([2.5, 1.5, 3.0]))

    def test_bct_candidate_keeps_frozen_lattice_cell(self):
        dynamic_structure = DynamicStructure(self.atoms_for_lattice(BCT(2.0, 3.0)))

        generated = dynamic_structure([2.0, 3.0])

        self.assert_cell_allclose(generated, BCT(2.0, 3.0).tocell().array)

    def test_mcl_right_angle_candidate_keeps_frozen_lattice_cell(self):
        lattice = MCL(1.0, 2.0, 3.0, 80.0)
        dynamic_structure = DynamicStructure(self.atoms_for_lattice(lattice))

        generated = dynamic_structure([1.0, 2.0, 3.0, 90.0])

        self.assert_cell_allclose(generated, lattice._cell(1.0, 2.0, 3.0, 90.0))

    def test_reference_structure_is_not_mutated(self):
        reference = self.atoms_for_lattice(ORC(1.0, 2.0, 3.0))
        original_cell = reference.cell.array.copy()
        original_positions = reference.get_positions().copy()

        DynamicStructure(reference)([2.0, 2.0, 3.0])

        np.testing.assert_allclose(reference.cell.array, original_cell, atol=1e-12)
        np.testing.assert_allclose(reference.get_positions(), original_positions, atol=1e-12)

    def test_wrong_vector_length_raises(self):
        dynamic_structure = DynamicStructure(self.atoms_for_lattice(ORC(1.0, 2.0, 3.0)))

        with self.assertRaises(ParameterVectorMismatchError):
            dynamic_structure([1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
