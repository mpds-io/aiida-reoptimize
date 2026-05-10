import sys

from aiida import load_profile
from aiida.engine import submit
from aiida.orm import Dict, Int, List, StructureData

from aiida_reoptimize.structure.MPDS_structure import get_geometry_MPDS
from aiida_reoptimize.workflows.Optimization.FleurSCF import (
    CDGFleurSCFOptimizer,
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
lattice = atoms.cell.get_bravais_lattice()
lattice_parameters = [float(lattice.vars()[name]) for name in lattice.parameters]
optimizer_parameters = {
    "itmax": Int(100),
    "structure": StructureData(ase=atoms),
    "parameters": Dict(
        {
            "algorithm_settings": {
                "tolerance": 1e-1,
                "learning_rate": 1e-2,
                "lr_increase": 1.2,
                "lr_decrease": 0.2,
                "delta": 0.0000529177,
            },
            "initial_parameters": List(list=lattice_parameters),
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

results = submit(CDGFleurSCFOptimizer, **optimizer_parameters)
print(f"Submitted CDGFleurSCFOptimizer: {results.pk}")
