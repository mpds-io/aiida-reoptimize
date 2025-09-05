from ...base.Extractors import BasicExtractor
from ...optimizers.convex.GD import (
    AdamOptimizer,
    ConjugateGradientOptimizer,
    RMSpropOptimizer,
)
from ...optimizers.convex.QN import BFGSOptimizer
from ...optimizers.PyMOO.PyMOO import PyMOO_Optimizer
from ..Evaluation.fleur_evaluators import FleurRelaxLatticeProblem


class BaseFleurRelaxOptimizer:
    evaluator_workchain = FleurRelaxLatticeProblem
    extractor = BasicExtractor(
        node_extractor=lambda x: x["output_relax_wc_para"]["energy"]
    )


class AdamFleurSCFOptimizer(BaseFleurRelaxOptimizer, AdamOptimizer):
    pass


class CDGFleurSCFOptimizer(BaseFleurRelaxOptimizer, ConjugateGradientOptimizer):
    pass


class RMSpropFleurSCFOptimizer(BaseFleurRelaxOptimizer, RMSpropOptimizer):
    pass


class BFGSFleurSCFOptimizer(BaseFleurRelaxOptimizer, BFGSOptimizer):
    pass


class PyMOOFleurSCFOptimizer(BaseFleurRelaxOptimizer, PyMOO_Optimizer):
    pass