from aiida import load_profile
from aiida.engine import submit
from aiida.orm import Dict, Int, StructureData
from ase.spacegroup import crystal

from aiida_reoptimize.workflows.Optimization.FleurSCF import (
    G3PCXFleurSCFOptimizer,
)

load_profile()

# Cubic SrTiO3, space group Pm-3m (221)
a = 3.905
atoms = crystal(
    ["Sr", "Ti", "O"],
    basis=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.0)],
    spacegroup=221,
    cellpar=[a, a, a, 90, 90, 90],
)

optimizer_parameters = {
    "itmax": Int(100),
    "structure": StructureData(ase=atoms),
    "parameters": Dict(
        {
            "algorithm_settings": {
                "pop_size": 20,
                "sampling": "LHS",
                "n_offsprings": 10,
                "n_parents": 3,
                "family_size": 2,
            },
            "calculator_parameters": {
                "codes": {
                    "inpgen": "inpgen@local_machine",
                    "fleur": "fleur@yascheduler",
                },
                "options": {
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 2,
                        "num_cores_per_mpiproc": 4,
                    },
                    "max_wallclock_seconds": 6 * 60 * 60,
                },
            },
        }
    ),
}

result = submit(G3PCXFleurSCFOptimizer, **optimizer_parameters)
print(f"Submitted G3PCXFleurSCFOptimizer: {result.pk}")
