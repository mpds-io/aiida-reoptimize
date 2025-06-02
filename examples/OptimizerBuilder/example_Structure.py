from aiida import load_profile
from aiida.common.exceptions import NotExistent
from aiida.engine import run
from aiida.orm import Dict, Int, List, load_node
from aiida_fleur.workflows.scf import FleurScfWorkChain
from ase.spacegroup import crystal

from aiida_reoptimize.base.OptimizerBuilder import OptimizerBuilder
from aiida_reoptimize.base.utils import find_nodes
from aiida_reoptimize.optimizers.convex.QN import BFGSOptimizer

load_profile()

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
a = 5.511
c = 7.796

atoms = crystal(
    ['Sr', 'Ti', 'O', 'O'],
    basis=[
        (0, 0, 0.25),
        (0.0, 0.5, 0.0),
        (0.2451, 0.7451, 0),
        (0, 0.5, 0.25)],
    spacegroup=140,
    cellpar=[a, a, c, 90, 90, 90],
)

# set up the calculator for structure optimization
builder = OptimizerBuilder.from_bulk(
    optimizer_workchain=BFGSOptimizer,
    calculator_workchain=FleurScfWorkChain,
    extractor=lambda x: x["output_scf_wc_para"]["total_energy"].value,
    calculator_parameters={"inpgen": inpgen_code, "fleur": fleur_code},
    bulk=atoms,
)

optimizer_parameters = {
    "itmax": Int(20),
    "parameters": Dict({
        "algorithm_settings": {"tolerance": 1e-8},
        "initial_parameters": List([a, c]),
    })
}

optimizer = builder.get_optimizer()
results = run(optimizer, **optimizer_parameters)