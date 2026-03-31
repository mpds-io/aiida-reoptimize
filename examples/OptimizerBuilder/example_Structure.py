from aiida import load_profile
from aiida.common.exceptions import NotExistent
from aiida.engine import run
from aiida.orm import Dict, Int, List, load_node
from aiida_fleur.workflows.scf import FleurScfWorkChain
from ase.spacegroup import crystal

from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.base.OptimizerBuilder import OptimizerBuilder
from aiida_reoptimize.base.utils import find_nodes
from aiida_reoptimize.optimizers.convex.GD import AdamOptimizer

load_profile()

dummy_extractor = BasicExtractor(node_extractor=lambda x: x["output_scf_wc_para"]["total_energy"])


def load_required_codes() -> dict[str, object]:
    """Load required AiiDA code nodes by configured labels."""

    fleur_label, inpgen_label = "fleur", "inpgen"
    nodes = find_nodes(fleur_label, inpgen_label)

    missing = [label for label in (fleur_label, inpgen_label) if label not in nodes]
    if missing:
        raise KeyError(f"Missing required code labels: {', '.join(missing)}")

    try:
        return {
            "fleur": load_node(nodes[fleur_label]),
            "inpgen": load_node(nodes[inpgen_label]),
        }
    except NotExistent as error:
        raise RuntimeError(f"Failed to load code node: {error}") from error


codes = load_required_codes()

# Setup structure
a = 5.511
c = 7.796

atoms = crystal(
    ["Sr", "Ti", "O", "O"],
    basis=[(0, 0, 0.25), (0.0, 0.5, 0.0), (0.2451, 0.7451, 0), (0, 0.5, 0.25)],
    spacegroup=140,
    cellpar=[a, a, c, 90, 90, 90],
)

# set up the calculator for structure optimization
builder = OptimizerBuilder.from_ase(
    optimizer_workchain=AdamOptimizer,
    calculator_workchain=FleurScfWorkChain,
    extractor=dummy_extractor,
    calculator_parameters={"inpgen": codes["inpgen"], "fleur": codes["fleur"]},
    bulk=atoms,
)

optimizer_parameters = {
    "itmax": Int(100),
    "parameters": Dict(
        {
            "algorithm_settings": {"tolerance": 1e-3},
            "initial_parameters": List(list=[a, c]),
        }
    ),
}

optimizer = builder.get_optimizer()
results = run(optimizer, **optimizer_parameters)
