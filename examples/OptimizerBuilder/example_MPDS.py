from aiida import load_profile
from aiida.common.exceptions import NotExistent
from aiida.engine import run
from aiida.orm import Dict, Int, List, load_node
from aiida_fleur.workflows.scf import FleurScfWorkChain

from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.base.OptimizerBuilder import OptimizerBuilder
from aiida_reoptimize.base.utils import find_nodes
from aiida_reoptimize.optimizers.convex.QN import BFGSOptimizer

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


# set up the calculator for structure optimization
builder = OptimizerBuilder.from_MPDS(
    optimizer_workchain=BFGSOptimizer,
    calculator_workchain=FleurScfWorkChain,
    extractor=dummy_extractor,
    calculator_parameters={"inpgen": codes["inpgen"], "fleur": codes["fleur"]},
    mpds_query="SrTiO3/221",
    structure_keyword=("structure",),
)

# Setup lattice parameters
# TODO find a better way to get these parameters
a = 3.905

optimizer_parameters = {
    "itmax": Int(100),
    "parameters": Dict(
        {
            "algorithm_settings": {"tolerance": 1e-3, "alpha": 0.1, "beta": 0.8},
            "initial_parameters": List(list=[a]),
        }
    ),
}

optimizer = builder.get_optimizer()
results = run(optimizer, **optimizer_parameters)
