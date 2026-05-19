import unittest

import numpy as np
from aiida import load_profile
from ase import Atoms
from ase.lattice import BCT, MCL, ORC

from aiida_reoptimize.structure.dynamic_structure import (
    DynamicStructure,
    ParameterVectorMismatchError,
    StructureCalculator,
    StructureGenerationError,
    _merge_magnetic_calc_parameters,
)
from aiida_reoptimize.structure.magmoms_utils import (
    MagneticMomentPreservationError,
    ase_to_structure_preserving_cell_and_magmoms,
)


def setUpModule():
    load_profile()


class BuilderWithCalcParameters:
    def __init__(self):
        self.calc_parameters = None
        self.structure = None


class BuilderWithoutCalcParameters:
    def __init__(self):
        self.structure = None


class CalculatorWithCalcParameters:
    @staticmethod
    def get_builder():
        return BuilderWithCalcParameters()


class CalculatorWithoutCalcParameters:
    @staticmethod
    def get_builder():
        return BuilderWithoutCalcParameters()


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

    def test_wrong_vector_length_error_includes_lattice_diagnostics(self):
        dynamic_structure = DynamicStructure(self.atoms_for_lattice(ORC(1.0, 2.0, 3.0)))

        with self.assertRaises(ParameterVectorMismatchError) as ctx:
            dynamic_structure([1.0, 2.0, 3.0, 4.0])

        message = str(ctx.exception)
        self.assertIn("Parameter vector length mismatch", message)
        self.assertIn("Structure generation diagnostics:", message)
        self.assertIn("expected_parameter_names: ('a', 'b', 'c')", message)
        self.assertIn("candidate_length: 4", message)
        self.assertIn("extra_value_count: 1", message)
        self.assertIn("<extra_3>=4.0", message)

    def test_structure_calculator_reports_target_index_on_vector_mismatch(self):
        reports = []
        structure_calculator = StructureCalculator(
            structure=self.atoms_for_lattice(ORC(1.0, 2.0, 3.0)),
            calculator=CalculatorWithCalcParameters,
            calculator_parameters={},
            reporter=reports.append,
        )

        with self.assertRaises(ParameterVectorMismatchError):
            structure_calculator.get_builder([1.0, 2.0, 3.0, 4.0], target_index=7)

        self.assertEqual(len(reports), 1)
        self.assertIn("Structure calculator diagnostics:", reports[0])
        self.assertIn("target_index: 7", reports[0])
        self.assertIn("calculator_workchain:", reports[0])
        self.assertIn("extra_value_count: 1", reports[0])

    def test_structure_input_path_error_includes_diagnostics(self):
        structure_calculator = StructureCalculator(
            structure=self.atoms_for_lattice(ORC(1.0, 2.0, 3.0)),
            calculator=CalculatorWithCalcParameters,
            calculator_parameters={},
            structure_keyword=("missing", "structure"),
        )

        with self.assertRaises(StructureGenerationError) as ctx:
            structure_calculator.get_builder([2.0, 2.0, 3.0], target_index=2)

        message = str(ctx.exception)
        self.assertIn("Cannot find 'missing' in structure input path", message)
        self.assertIn("stage: 'structure_input_injection'", message)
        self.assertIn("target_index: 2", message)

    def test_ase_initial_magmoms_survive_dynamic_structure_generation(self):
        reference = Atoms(
            "Fe2",
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=ORC(1.0, 2.0, 3.0).tocell(),
            pbc=True,
        )
        reference.set_initial_magnetic_moments([1.0, -1.0])

        generated = DynamicStructure(reference)([2.0, 2.0, 3.0])

        self.assertIn("initial_magmoms", generated.arrays)
        np.testing.assert_allclose(generated.get_initial_magnetic_moments(), [1.0, -1.0])

    def test_magnetic_structure_conversion_preserves_cell_positions_and_kinds(self):
        atoms = Atoms(
            "Fe2",
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=np.diag([2.5, 1.5, 3.0]),
            pbc=True,
        )
        atoms.set_initial_magnetic_moments([1.0, -1.0])

        structure, calc_parameters, kind_magmoms = ase_to_structure_preserving_cell_and_magmoms(atoms)

        np.testing.assert_allclose(structure.cell, atoms.cell.array, atol=1e-12)
        np.testing.assert_allclose([site.position for site in structure.sites], atoms.get_positions(), atol=1e-12)
        self.assertEqual([site.kind_name for site in structure.sites], ["Fe1", "Fe2"])
        self.assertEqual(kind_magmoms, {"Fe1": 1.0, "Fe2": -1.0})
        self.assertEqual(calc_parameters["comp"], {"jspins": 2})
        self.assertEqual(calc_parameters["atom_magmom_Fe1"]["id"], 26.1)
        self.assertEqual(calc_parameters["atom_magmom_Fe1"]["bmu"], 1.0)
        self.assertEqual(calc_parameters["atom_magmom_Fe2"]["id"], 26.2)
        self.assertEqual(calc_parameters["atom_magmom_Fe2"]["bmu"], -1.0)

    def test_zero_initial_magmoms_do_not_require_magnetic_calc_parameters(self):
        atoms = Atoms(
            "Fe2",
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=np.diag([2.0, 2.0, 3.0]),
            pbc=True,
        )
        atoms.set_initial_magnetic_moments([0.0, 0.0])

        _, calc_parameters, kind_magmoms = ase_to_structure_preserving_cell_and_magmoms(atoms)

        self.assertEqual(calc_parameters, {})
        self.assertEqual(kind_magmoms, {"Fe1": 0.0})

    def test_magnetic_calc_parameter_merge_preserves_user_atom_settings(self):
        merged = _merge_magnetic_calc_parameters(
            {
                "atom": {"element": "Fe", "rmt": 2.2, "lmax": 8},
                "comp": {"kmax": 3.7},
            },
            {
                "atom_magmom_Fe1": {"element": "Fe", "id": 26.1, "bmu": 1.0},
                "comp": {"jspins": 2},
            },
        )

        self.assertEqual(merged["comp"], {"kmax": 3.7, "jspins": 2})
        self.assertEqual(
            merged["atom_magmom_Fe1"],
            {"element": "Fe", "rmt": 2.2, "lmax": 8, "id": 26.1, "bmu": 1.0},
        )

    def test_structure_calculator_injects_magnetic_calc_parameters(self):
        atoms = Atoms(
            "Fe2",
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=ORC(1.0, 2.0, 3.0).tocell(),
            pbc=True,
        )
        atoms.set_initial_magnetic_moments([1.0, -1.0])
        structure_calculator = StructureCalculator(
            structure=atoms,
            calculator=CalculatorWithCalcParameters,
            calculator_parameters={"calc_parameters": {"atom": {"element": "Fe", "jri": 981}}},
        )

        builder = structure_calculator.get_builder([2.0, 2.0, 3.0])
        calc_parameters = builder.calc_parameters.get_dict()

        np.testing.assert_allclose(builder.structure.cell, np.diag([2.0, 2.0, 3.0]), atol=1e-12)
        self.assertEqual([site.kind_name for site in builder.structure.sites], ["Fe1", "Fe2"])
        self.assertEqual(calc_parameters["atom_magmom_Fe1"]["jri"], 981)
        self.assertEqual(calc_parameters["atom_magmom_Fe1"]["bmu"], 1.0)
        self.assertEqual(calc_parameters["atom_magmom_Fe2"]["bmu"], -1.0)
        self.assertEqual(calc_parameters["comp"], {"jspins": 2})

    def test_noncollinear_initial_magmoms_raise(self):
        atoms = Atoms(
            "Fe2",
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=np.diag([2.0, 2.0, 3.0]),
            pbc=True,
        )
        atoms.set_initial_magnetic_moments([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

        with self.assertRaises(MagneticMomentPreservationError):
            ase_to_structure_preserving_cell_and_magmoms(atoms)

    def test_magnetic_structure_requires_calc_parameters_port(self):
        atoms = Atoms(
            "Fe2",
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=ORC(1.0, 2.0, 3.0).tocell(),
            pbc=True,
        )
        atoms.set_initial_magnetic_moments([1.0, -1.0])
        structure_calculator = StructureCalculator(
            structure=atoms,
            calculator=CalculatorWithoutCalcParameters,
            calculator_parameters={},
        )

        with self.assertRaises(MagneticMomentPreservationError):
            structure_calculator.get_builder([2.0, 2.0, 3.0])

    def test_orc_to_tet_symmetry_change_generates_valid_structure(self):
        dynamic_structure = DynamicStructure(self.atoms_for_lattice(ORC(1.0, 2.0, 3.0)))

        generated = dynamic_structure([2.0, 2.0, 3.0])

        self.assert_cell_allclose(generated, np.diag([2.0, 2.0, 3.0]))

    def test_orc_to_cub_symmetry_change_generates_valid_structure(self):
        dynamic_structure = DynamicStructure(self.atoms_for_lattice(ORC(1.0, 2.0, 3.0)))

        generated = dynamic_structure([2.0, 2.0, 2.0])

        self.assert_cell_allclose(generated, np.diag([2.0, 2.0, 2.0]))

    def test_mcl_to_orc_symmetry_change_generates_valid_structure(self):
        lattice = MCL(1.0, 2.0, 3.0, 80.0)
        dynamic_structure = DynamicStructure(self.atoms_for_lattice(lattice))

        generated = dynamic_structure([1.0, 2.0, 3.0, 90.0])

        self.assert_cell_allclose(generated, lattice._cell(1.0, 2.0, 3.0, 90.0))


if __name__ == "__main__":
    unittest.main()
