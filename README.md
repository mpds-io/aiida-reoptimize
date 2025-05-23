# Atomic structure optimization backed by AiiDA


## Technical details

The external PyMOO library is used. The PyMOO optimizer needs two specific classes: Algorithm and Problem. These classes cannot be passed through the AiiDA context, because they cannot be serialized. Therefore, they are set as the local variables. Calculations are now run by a separate __evaluator__ workflow. The evaluator workflow class accept the AiiDA `WorkChain` for the problem given, the list of parameters for this problem to be performed, the extractor function to extract the data, and the penalty, i.e. the value for the calculation that failed. In total, they are passed to the optimizer class to run the calculation given and collect the data resulted.

There are two types of the optimizer classes in total. The first simply transfers the data to the workflow, and the second compiles a new workflow through builder. It is used to modify the structure, accepting the new lattice parameters and counting WC with the updated structure. This evaluator is made to work with the `StructureCalculator` class. This class accepts a calculated `WorkChain` (e.g. `fleurSCF`), the `ASE` structure, and the calculated parameters, so that the `get_builder(x)` method returns a new workflow for calculating the same structure but with the `x` parameters.


## References

- [pymoo: Multi-objective Optimization in Python](https://pymoo.org)
- [similar work by Dominik Greschm ](https://github.com/greschd/aiida-optimize)


## License

MIT

Copyright (c) 2025 Tilde Materials Informatics and Materials Platform for Data Science LLC
