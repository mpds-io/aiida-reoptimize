from aiida import load_profile
from aiida.common.exceptions import NotExistent
from aiida.engine import run
from aiida.orm import Dict, Int, List, load_node
from aiida_fleur.workflows.relax import FleurRelaxWorkChain

from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.base.OptimizerBuilder import OptimizerBuilder
from aiida_reoptimize.base.utils import find_nodes
from aiida_reoptimize.optimizers.convex.GD import AdamOptimizer

load_profile()

dummy_extractor = BasicExtractor(node_extractor=lambda x: x["output_relax_wc_para"]["energy"])


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

# FleurRelax parameters
wf_relax_scf = {
    "fleur_runmax": 4,
    "itmax_per_run": 70,
    "mode": "energy",
    "energy_converged": 1e-3,
}

wf_relax = {
    "force_criterion": 0.01,
    "relax_iter": 8,
}


# set up the calculator for structure optimization
builder = OptimizerBuilder.from_MPDS(
    optimizer_workchain=AdamOptimizer,
    calculator_workchain=FleurRelaxWorkChain,
    extractor=dummy_extractor,
    mpds_query="SrTiO3/140",
    calculator_parameters={
        "scf": {
            "inpgen": codes["inpgen"],
            "fleur": codes["fleur"],
            "wf_parameters": wf_relax_scf,
        },
        "wf_parameters": wf_relax,
    },
    structure_keyword=("scf", "structure"),
)

# Setup lattice parameters
# TODO find a better way to get these parameters
a = 5.511
c = 7.796

optimizer_parameters = {
    "itmax": Int(2),
    "parameters": Dict(
        {
            "algorithm_settings": {"tolerance": 1e-3},
            "initial_parameters": List(list=[a, c]),
        }
    ),
}

optimizer = builder.get_optimizer()
results = run(optimizer, **optimizer_parameters)
