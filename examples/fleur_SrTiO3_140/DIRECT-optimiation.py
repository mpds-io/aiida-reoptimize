import sys

sys.path.append("/root/code/aiida-reoptimize")

from aiida import load_profile
from aiida.engine import ToContext, run
from aiida.orm import Float, Int, List, Str

from src.optimizers.DIRECT import DIRECTWorkChain
from src.problems.SrTiO3_140_fleur import ObjectiveFunctionWorkChain

load_profile()


class ExampleDIRECT(DIRECTWorkChain):
    def evaluate(self):
        """Execute and track parallel objective function evaluations.

        Submits FleurRelaxWorkChain instances for all particle positions using AiiDA's process tracking.
        Uses ToContext to synchronize workflow execution.
        """  # noqa: E501

        target_values = {}
        # Get objective function work chain class
        ObjectiveWorkChain = ObjectiveFunctionWorkChain
        for i, pos in enumerate(self.ctx.targets):
            position_list = List(list=[float(p) for p in pos])
            future = self.submit(ObjectiveWorkChain, x=position_list)
            target_values[f"eval_{i}"] = future
        return ToContext(**target_values)  # Wait for all processes to complete


# Run the optimization
results = run(
    ExampleDIRECT,
    bounds=List(list=[[5.207, 5.807], [7.696, 8.296]]),
    dimensions=Int(2),
    max_iterations=Int(25),
    key_value=Str("energy"),
    epsilon=Float(1e-3),
)

print("Optimization Results:")
print(f"Best position: {results['optimized_parameters']}")
print(f"Best energy: {results['final_value']}")
