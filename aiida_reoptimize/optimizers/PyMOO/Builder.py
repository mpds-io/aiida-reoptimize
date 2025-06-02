from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX

# TERMINATION
from pymoo.core.termination import NoTermination

# SAMPLING
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS

# SELECTION
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection

# CROSSOVER
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.pntx import PointCrossover, SinglePointCrossover, TwoPointCrossover  # noqa: E501
from pymoo.operators.crossover.expx import ExponentialCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.hux import HalfUniformCrossover

# MUTATION
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PolynomialMutation

# REPAIR
from pymoo.core.repair import Repair


class AlgorithmBuilder:

    SAMPLING = {
        "FRS": FloatRandomSampling,
        "LHS": LHS,
    }
    SELECTION = {
        "RND": RandomSelection,
        "TOS": TournamentSelection,
    }
    CROSSOVER = {
        "SBX": SBX,
        "PNTX": PointCrossover,
        "SPCX": SinglePointCrossover,
        "TPCX": TwoPointCrossover,
        "EXPCX": ExponentialCrossover,
        "UX": UniformCrossover,
        "HUX": HalfUniformCrossover,
    }
    MUTATION = {
        "BITFLIP": BitflipMutation,
        "PM": PolynomialMutation,
    }

    # The REPAIR dictionary is intentionally left empty for future extensions
    # or custom repair operators.
    REPAIR = {}

    all_operators = {
        "sampling": SAMPLING,
        "selection": SELECTION,
        "crossover": CROSSOVER,
        "mutation": MUTATION,
        "repair": REPAIR,
    }

    allowed_keyords_map = {
        "DE": ["pop_size", "n_offsprings", "sampling", "variant"],

        "ES": ["pop_size", "n_offsprings", "rule", "phi", "gamma", "sampling"],

        "GA": ["pop_size", "sampling", "selection",
               "crossover", "mutation", "eliminate_duplicates",
               "n_offsprings"],

        "G3PCX": ["pop_size", "sampling", "n_offsprings",
                  "n_parents", "family_size", "repair"],

        "PSO": ["pop_size", "sampling", "w", "c1", "c2",
                "adaptive", "initial_velocity",
                "max_velocity_rate", "pertube_best"]
    }

    # TODO add cmaes
    ALGORITHMS = {
            "DE": DE,
            "ES": ES,
            "GA": GA,
            "G3PCX": G3PCX,
            "PSO": PSO,
        }

    @staticmethod
    def __process_kwargs(algorithm_name: str, **kwargs):
        # Check if the algorithm name is valid
        allowed_kwargs = AlgorithmBuilder.allowed_keyords_map.get(algorithm_name, [])  # noqa: E501
        if not allowed_kwargs:
            raise ValueError(f"Algorithm {algorithm_name} is not supported.")

        parameters = {}
        for key in kwargs:
            if key not in allowed_kwargs:
                raise ValueError(f"Invalid keyword argument: {key}")
            # If it is sampling or etc. then it check special dictionary
            # otherwise it is a normal parameter
            if key in AlgorithmBuilder.all_operators:
                if kwargs[key] in AlgorithmBuilder.all_operators[key]:
                    parameters[key] = AlgorithmBuilder.all_operators[key][kwargs[key]]  # noqa: E501
            else:
                parameters[key] = kwargs[key]

        return parameters

    @staticmethod
    def build_algorithm(algorithm_name: str, **kwargs):
        """
        Factory method to create an algorithm class based on the name.

        Args:
            algorithm_name (str): Name of the algorithm.
            **kwargs: Additional arguments for the algorithm.
        """

        if algorithm_name not in AlgorithmBuilder.ALGORITHMS:
            raise ValueError(f"Algorithm {algorithm_name} is not supported.")

        keywords = {
            "pop_size": 100,
            "termination": NoTermination()
        }

        keywords.update(AlgorithmBuilder.__process_kwargs(algorithm_name, **kwargs))  # noqa: E501
        algorithm_class = AlgorithmBuilder.ALGORITHMS[algorithm_name]
        return algorithm_class(**keywords)

