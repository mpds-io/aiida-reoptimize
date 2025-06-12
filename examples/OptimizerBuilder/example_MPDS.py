from aiida import load_profile
from aiida.common.exceptions import NotExistent
from aiida.engine import run
from aiida.orm import Dict, Int, List, load_node
from aiida_fleur.workflows.scf import FleurScfWorkChain

from aiida_reoptimize.base.Extratractors import BasicExtractor
from aiida_reoptimize.base.OptimizerBuilder import OptimizerBuilder
from aiida_reoptimize.base.utils import find_nodes
from aiida_reoptimize.optimizers.convex.QN import BFGSOptimizer

load_profile()

dummy_extractor = BasicExtractor(
    node_exctractor=lambda x: x["output_scf_wc_para"]["total_energy"]
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


# set up the calculator for structure optimization
builder = OptimizerBuilder.from_MPDS(
    optimizer_workchain=BFGSOptimizer,
    calculator_workchain=FleurScfWorkChain,
    extractor=dummy_extractor,
    calculator_parameters={"inpgen": inpgen_code, "fleur": fleur_code},
    mpds_query="SrTiO3/140",
)

# Setup lattice parameters
a = 5.511
c = 7.796

optimizer_parameters = {
    "itmax": Int(20),
    "parameters": Dict({
        "algorithm_settings": {"tolerance": 1e-3},
        "initial_parameters": List([a, c]),
    }),
}

optimizer = builder.get_optimizer()
results = run(optimizer, **optimizer_parameters)