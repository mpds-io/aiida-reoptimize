from aiida import load_profile
from aiida.common.exceptions import NotExistent
from aiida.engine import submit
from aiida.orm import Dict, Int, List, StructureData, load_node
from ase.spacegroup import crystal

from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.base.utils import find_nodes
from aiida_reoptimize.workflows.Optimization.FleurSCF import (
    CDGFleurSCFOptimizer,
)

load_profile()

dummy_extractor = BasicExtractor(
    node_extractor=lambda x: x["output_scf_wc_para"]["total_energy"]
)

# Find aiida codes for Fleur and inpgen
fleur_node_label, inpgen_node_label = "fleur", "inpgen"
nodes = find_nodes(fleur_node_label, inpgen_node_label)
required_codes = [fleur_node_label, inpgen_node_label]

for code_label in required_codes:
    if code_label not in nodes:
        raise KeyError(f"Missing required code: {code_label}")

    try:
        fleur_code = load_node(nodes[fleur_node_label])
        inpgen_code = load_node(nodes[inpgen_node_label])
    except NotExistent as e:
        raise RuntimeError(f"Failed to load code node: {e}") from e

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
    "itmax": Int(100),
    "structure": StructureData(ase=atoms),
    "parameters": Dict({
        "algorithm_settings": {
            "tolerance": 1e-1,
            "learning_rate": 1e-2,
            "lr_increase": 1.2,
            "lr_decrease": 0.2,
            "delta": 0.0000529177,
        },
        "initial_parameters": List([a, c]),
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
    }),
}

results = submit(CDGFleurSCFOptimizer, **optimizer_parameters)
print(f"Submitted AdamFleurSCFOptimizer: {results.pk}")
