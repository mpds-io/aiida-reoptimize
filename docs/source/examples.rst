Examples
========

The ``examples/`` directory contains runnable scripts demonstrating different
use cases for ``aiida-reoptimize``.

OptimizerBuilder
-----------------

Dynamic workflow examples showing how to assemble optimizers at runtime.

+------------------------------------------+------------------------------------------------------+
| File                                     | Description                                          |
+==========================================+======================================================+
| ``example_Problem.py``                   | Build an optimizer from a simple function problem    |
+------------------------------------------+------------------------------------------------------+
| ``example_Structure.py``                 | Build an optimizer from an ASE structure              |
+------------------------------------------+------------------------------------------------------+
| ``example_MPDS.py``                      | Build an optimizer from an MPDS database query       |
+------------------------------------------+------------------------------------------------------+

Dummy Problems
--------------

Toy problem examples that test each optimizer algorithm without a real
DFT calculation.

+------------------------------------------+------------------------------------------------------+
| File                                     | Description                                          |
+==========================================+======================================================+
| ``Dummy_user_problem.py``                | Custom user-defined optimization problem             |
+------------------------------------------+------------------------------------------------------+
| ``dummy_Adam.py``                        | Adam optimizer on a simple problem                   |
+------------------------------------------+------------------------------------------------------+
| ``dummy_BFGS.py``                        | BFGS optimizer on a simple problem                    |
+------------------------------------------+------------------------------------------------------+
| ``dummy_CGD.py``                         | Conjugate gradient optimizer on a simple problem     |
+------------------------------------------+------------------------------------------------------+
| ``dummy_RMSProp.py``                     | RMSprop optimizer on a simple problem                 |
+------------------------------------------+------------------------------------------------------+
| ``dummy_all_convex_x2.py``              | Comparison of all gradient-based optimizers on x²    |
+------------------------------------------+------------------------------------------------------+
| ``dummy_PyMOO_G3PCX.py``                | PyMOO G3PCX algorithm on a simple problem            |
+------------------------------------------+------------------------------------------------------+
| ``dummy_PyMOO_NRBO.py``                 | PyMOO NRBO algorithm on a simple problem             |
+------------------------------------------+------------------------------------------------------+
| ``dummy_PyMOO_PSO.py``                  | PyMOO PSO algorithm on a simple problem              |
+------------------------------------------+------------------------------------------------------+

Static Workflows
-----------------

Pre-registered workchain examples for production use with ``submit()``.

+------------------------------------------+------------------------------------------------------+
| File                                     | Description                                          |
+==========================================+======================================================+
| ``adam_crystal_test.py``                 | Adam optimizer with CRYSTAL                          |
+------------------------------------------+------------------------------------------------------+
| ``adam_fleur_test.py``                  | Adam optimizer with FLEUR SCF                        |
+------------------------------------------+------------------------------------------------------+
| ``CDG_crystal_test.py``                 | Conjugate gradient with CRYSTAL                      |
+------------------------------------------+------------------------------------------------------+
| ``CDG_fleur_test.py``                   | Conjugate gradient with FLEUR SCF                    |
+------------------------------------------+------------------------------------------------------+
| ``CDG_fleur_mpds.py``                   | Conjugate gradient with FLEUR and MPDS structure     |
+------------------------------------------+------------------------------------------------------+
| ``g3pcx_fleur_srtio3_221.py``           | G3PCX optimizer with FLEUR on SrTiO₃                |
+------------------------------------------+------------------------------------------------------+
| ``g3px_fleur_mpds.py``                  | G3PCX optimizer with FLEUR and MPDS structure        |
+------------------------------------------+------------------------------------------------------+
| ``nrbo_fleur_mpds.py``                  | NRBO optimizer with FLEUR and MPDS structure         |
+------------------------------------------+------------------------------------------------------+
| ``nrbo_fleur_srtio3_221.py``            | NRBO optimizer with FLEUR on SrTiO₃                  |
+------------------------------------------+------------------------------------------------------+

Targeted Examples
------------------

+------------------------------------------+------------------------------------------------------+
| Directory / File                         | Description                                          |
+==========================================+======================================================+
| ``Crystal/example_Crystal.py``           | CRYSTAL lattice optimization example                 |
+------------------------------------------+------------------------------------------------------+
| ``Evaluators/Au_fleur.py``               | FLEUR evaluator example on gold                      |
+------------------------------------------+------------------------------------------------------+
| ``fleur_SrTiO3_140/``                    | FLEUR optimization on SrTiO₃ (space group 140)      |
+------------------------------------------+------------------------------------------------------+
| ``fleur_relax/adam.py``                  | FLEUR relax optimization with Adam                   |
+------------------------------------------+------------------------------------------------------+
