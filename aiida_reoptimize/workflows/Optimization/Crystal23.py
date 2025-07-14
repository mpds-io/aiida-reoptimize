from ...base.Extractors import BasicExtractor
from ...optimizers.convex.GD import AdamOptimizer, RMSpropOptimizer
from ...optimizers.convex.QN import BFGSOptimizer
from ...optimizers.PyMOO.PyMOO import PyMOO_Optimizer
from ..Evaluation.crystal23_evaluation import CrystalLatticeProblem


class BaseCrystal23Optimizer:
    evaluator_workchain = CrystalLatticeProblem
    extractor = BasicExtractor(
        node_extractor=lambda x: x["output_parameters"]["energy"]
    )


class AdamCrystal23Optimizer(BaseCrystal23Optimizer, AdamOptimizer):
    pass


class RMSpropCrystal23Optimizer(BaseCrystal23Optimizer, RMSpropOptimizer):
    pass


class BFGSFCrystal23Optimizer(BaseCrystal23Optimizer, BFGSOptimizer):
    pass


class PyMOOCrystal23Optimizer(BaseCrystal23Optimizer, PyMOO_Optimizer):
    pass