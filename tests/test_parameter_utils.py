import unittest

import numpy as np
from ase import Atoms
from ase.lattice import ORC

from aiida_reoptimize.optimizers.parameter_utils import (
    OptimizationParameterError,
    prepare_optimization_parameters,
)


class TestParameterUtils(unittest.TestCase):
    @staticmethod
    def atoms_for_lattice(lattice):
        return Atoms("Mg", scaled_positions=[[0.0, 0.0, 0.0]], cell=lattice.tocell(), pbc=True)

    def test_mismatched_initial_parameters_use_structure_inferred_values(self):
        reports = []
        atoms = self.atoms_for_lattice(ORC(1.0, 2.0, 3.0))

        normalized = prepare_optimization_parameters(
            {"initial_parameters": [1.0, 2.0, 3.0, 90.0]},
            structure=atoms,
            require_bounds=False,
            require_initial_parameters=True,
            reporter=reports.append,
        )

        self.assertEqual(normalized["dimensions"], 3)
        np.testing.assert_allclose(normalized["initial_parameters"], [1.0, 2.0, 3.0])
        self.assertEqual(len(reports), 1)
        self.assertIn("Using structure-inferred initial_parameters instead", reports[0])
        self.assertIn("expected_parameter_names: ('a', 'b', 'c')", reports[0])
        self.assertIn("provided_count: 4", reports[0])
        self.assertIn("<extra_3>=90.0", reports[0])

    def test_bounds_mismatch_error_includes_lattice_diagnostics(self):
        reports = []
        atoms = self.atoms_for_lattice(ORC(1.0, 2.0, 3.0))

        with self.assertRaises(OptimizationParameterError) as ctx:
            prepare_optimization_parameters(
                {
                    "initial_parameters": [1.0, 2.0, 3.0, 90.0],
                    "bounds": [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [80.0, 100.0]],
                },
                structure=atoms,
                require_bounds=True,
                require_initial_parameters=False,
                reporter=reports.append,
            )

        message = str(ctx.exception)
        self.assertIn("Length mismatch: 'bounds' contains 4 entries but 3 parameters were inferred", message)
        self.assertIn("Optimization parameter diagnostics:", message)
        self.assertIn("lattice_name: 'ORC'", message)
        self.assertIn("expected_parameter_names: ('a', 'b', 'c')", message)
        self.assertEqual(len(reports), 1)

    def test_initial_parameters_are_inferred_from_structure_when_missing(self):
        atoms = self.atoms_for_lattice(ORC(1.0, 2.0, 3.0))

        normalized = prepare_optimization_parameters(
            {},
            structure=atoms,
            require_bounds=False,
            require_initial_parameters=True,
        )

        self.assertEqual(normalized["dimensions"], 3)
        np.testing.assert_allclose(normalized["initial_parameters"], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
