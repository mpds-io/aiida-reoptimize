from aiida import load_profile
from aiida.engine import submit
from aiida.orm import Dict, Int, List, StructureData
from ase.spacegroup import crystal

from aiida_reoptimize.workflows.Optimization.Crystal import (
    CDGCrystalOptimizer,
)

load_profile()

calculation_settings = {
    "label": "Crystal SinglePoint",
    "scf": {
        "k_points": [8, 16],
        "dft": {
            "SPIN": False,
            "xc": "PBE0",
            "grid": "XLGRID",
            "numerical": {"TOLLDENS": 8, "TOLLGRID": 16},
        },
        "numerical": {
            "TOLDEE": 6,
            "BIPOSIZE": 256000000,
            "EXCHSIZE": 256000000,
            "MAXCYCLE": 200,
        },
    },
}

basis_name = "MPDSBSL_NEUTRAL_24"

options = {
    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 8},
    "try_oxi_if_fails": False,
}

# Setup structure
a = 5.51
c = 7.81

atoms = crystal(
    ["Sr", "Ti", "O", "O"],
    basis=[(0, 0, 0.25), (0.0, 0.5, 0.0), (0.2451, 0.7451, 0), (0, 0.5, 0.25)],
    spacegroup=140,
    cellpar=[a, a, c, 90, 90, 90],
)

optimizer_parameters = {
    "itmax": Int(10),
    "structure": StructureData(ase=atoms),
    "parameters": Dict({
        "algorithm_settings": {
            "tolerance": 1e-2,
            "learning_rate": 1e-2,
            "lr_increase": 1.2,
        },
        "initial_parameters": List([a, c]),
        "calculator_parameters": {
            "codes": {"code": "Pcrystal23@yascheduler"},
            "parameters": calculation_settings,
            "basis_family": basis_name,
            "options": options,
        },
    }),
}

results = submit(CDGCrystalOptimizer, **optimizer_parameters)
print(f"Submitted CDGCrystalOptimizer: {results.pk}")