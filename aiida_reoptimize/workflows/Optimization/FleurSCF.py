from ...base.Extractors import BasicExtractor
from ...optimizers.convex.GD import (
    AdamOptimizer,
    ConjugateGradientOptimizer,
    RMSpropOptimizer,
)
from ...optimizers.convex.QN import BFGSOptimizer
from ...optimizers.PyMOO.PyMOO import PyMOO_Optimizer
from ..Evaluation.fleur_evaluators import FleurSCFLatticeProblem


class BaseFleurSCFOptimizer:
    evaluator_workchain = FleurSCFLatticeProblem
    extractor = BasicExtractor(
        node_extractor=lambda x: x["output_scf_wc_para"]["total_energy"]
    )


class AdamFleurSCFOptimizer(BaseFleurSCFOptimizer, AdamOptimizer):
    pass


class CDGFleurSCFOptimizer(BaseFleurSCFOptimizer, ConjugateGradientOptimizer):
    pass


class RMSpropFleurSCFOptimizer(BaseFleurSCFOptimizer, RMSpropOptimizer):
    pass


class BFGSFleurSCFOptimizer(BaseFleurSCFOptimizer, BFGSOptimizer):
    pass


class PyMOOFleurSCFOptimizer(BaseFleurSCFOptimizer, PyMOO_Optimizer):
    pass
