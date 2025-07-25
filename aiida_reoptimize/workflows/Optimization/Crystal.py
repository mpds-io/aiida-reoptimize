from ...base.Extractors import BasicExtractor
from ...optimizers.convex.GD import (
    AdamOptimizer,
    ConjugateGradientOptimizer,
    RMSpropOptimizer,
)
from ...optimizers.convex.QN import BFGSOptimizer
from ...optimizers.PyMOO.PyMOO import PyMOO_Optimizer
from ..Evaluation.crystal_evaluation import CrystalLatticeProblem


class BaseCrystalOptimizer:
    evaluator_workchain = CrystalLatticeProblem
    extractor = BasicExtractor(
        node_extractor=lambda x: x["output_parameters"]["energy"]
    )


class AdamCrystalOptimizer(BaseCrystalOptimizer, AdamOptimizer):
    pass


class CDGCrystalOptimizer(BaseCrystalOptimizer, ConjugateGradientOptimizer):
    pass


class RMSpropCrystalOptimizer(BaseCrystalOptimizer, RMSpropOptimizer):
    pass


class BFGSFCrystalOptimizer(BaseCrystalOptimizer, BFGSOptimizer):
    pass


class PyMOOCrystalOptimizer(BaseCrystalOptimizer, PyMOO_Optimizer):
    pass