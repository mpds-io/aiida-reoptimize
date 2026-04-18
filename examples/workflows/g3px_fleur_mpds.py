import sys

from aiida import load_profile
from aiida.engine import submit
from aiida.orm import Dict, Int, StructureData

from aiida_reoptimize.structure.MPDS_structure import get_geometry_MPDS
from aiida_reoptimize.workflows.Optimization.FleurSCF import (
    G3PCXFleurSCFOptimizer,
)

load_profile()

try:
    phase = sys.argv[1].split("/")
except IndexError:
    phase = ("MgO", "225")
    print("Default phase for testing: " + "/".join(phase))

if len(phase) == 3:
    formula, sgs, pearson = phase
else:
    formula, sgs, pearson = phase[0], phase[1], None

sgs = int(sgs)

atoms = get_geometry_MPDS(({"formulae": formula, "sgs": sgs}))

optimizer_parameters = {
    "itmax": Int(100),
    "structure": StructureData(ase=atoms),
    "parameters": Dict(
        {
            "bounds": 0.2,
            "algorithm_settings": {
                "pop_size": 20,
                "tol": 1e-3,
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
