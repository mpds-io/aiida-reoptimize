from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, Int, List, load_code
from aiida.plugins import DataFactory
from aiida_crystal_dft.workflows.base import BaseCrystalWorkChain

from aiida_reoptimize.base.Extractors import BasicExtractor
from aiida_reoptimize.base.OptimizerBuilder import OptimizerBuilder
from aiida_reoptimize.optimizers.convex.GD import AdamOptimizer

load_profile()

calculation_settings = Dict(
    dict={
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
                "TOLDEE": 9,
                "BIPOSIZE": 256000000,
                "EXCHSIZE": 256000000,
                "MAXCYCLE": 400,
        }}}
)

basis_family, _ = DataFactory("crystal_dft.basis_family").get_or_create(
    "MPDSBSL_NEUTRAL_24"
)  # noqa: E501

options = DataFactory("dict")(
    dict={
        "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 8},
        "try_oxi_if_fails": False,
    }
)

inputs = {
    "code": load_code("Pcrystal23@yascheduler"),
    "parameters": calculation_settings,
    "basis_family": basis_family,
    "options": options,
}

dummy_extractor = BasicExtractor(
    node_extractor=lambda x: x["output_parameters"]["energy"]
)
# 
builder = OptimizerBuilder.from_MPDS(
    optimizer_workchain=AdamOptimizer,
    calculator_workchain=BaseCrystalWorkChain,
    extractor=dummy_extractor,
    calculator_parameters=inputs,
    mpds_query="SrTiO3/221",
)
# 
# Setup lattice parameters
# TODO find a better way to get these parameters
a = 3.905
# 
optimizer_parameters = {
    "itmax": Int(100),
    "parameters": Dict({
        "algorithm_settings": {"tolerance": 1e-3, "learning_rate": 0.05},
        "initial_parameters": List([a]),
    }),
}
# 
optimizer = builder.get_optimizer()
results = run(optimizer, **optimizer_parameters)