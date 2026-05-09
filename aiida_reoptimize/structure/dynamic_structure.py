from typing import Type

import ase
from aiida.engine import WorkChain
from aiida.orm import StructureData


class ParameterVectorMismatchError(ValueError):
    """Raised when a target vector does not match the lattice parameterization."""


class DynamicStructure:
    """Generate new structures from a given ASE Atoms object by changing cell parameters.

    Given a reference structure, calling this object with a parameter vector ``x``
    creates a new structure whose unit cell is constructed from ``x`` using the
    Bravais lattice class. Atomic positions are scaled to fit the new cell.
    """

    def __init__(self, structure):
        self.__structure = structure
        self.__structure_lattice = structure.cell.get_bravais_lattice()
        self.__parameter_names = tuple(self.__structure_lattice.parameters)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the ASE Bravais lattice parameter names expected by ``__call__``."""

        return self.__parameter_names

    def initial_parameters(self) -> list[float]:
        """Return the current lattice parameters in the same order expected by ``__call__``."""

        values = self.__structure_lattice.vars()
        return [float(values[name]) for name in self.__parameter_names]

    def __call__(self, x):
        """Create a new ASE Atoms object with cell parameters given by ``x``.

        Args:
            x: Parameter vector matching the free parameters of the Bravais lattice.

        Returns:
            A new ASE Atoms object with the updated cell and scaled positions.
        """
        parameters = list(x)
        if len(parameters) != len(self.__parameter_names):
            lattice_name = getattr(
                self.__structure_lattice,
                "name",
                self.__structure_lattice.__class__.__name__,
            )
            expected = ", ".join(self.__parameter_names)
            raise ParameterVectorMismatchError(
                f"Parameter vector length mismatch for {lattice_name}: "
                f"expected {len(self.__parameter_names)} values ({expected}), "
                f"got {len(parameters)}."
            )

        new_cell = self.__structure_lattice.__class__(**dict(zip(self.__parameter_names, parameters, strict=True)))
        new_structure = self.__structure.copy()
        new_structure.set_cell(new_cell.tocell(), scale_atoms=True)
        return new_structure


class StructureCalculator:
    """Combine a ``DynamicStructure`` with a calculator WorkChain.

    For each parameter vector ``x``, generates a distorted structure and
    returns a ready-to-submit process builder with the structure and
    calculator parameters inserted.
    """

    def __init__(
        self,
        structure: ase.atoms.Atoms,
        calculator: Type[WorkChain],
        calculator_parameters: dict,
        structure_keyword: tuple = ("structure",),
    ):
        self.structure = structure
        self.calculator = calculator
        self.parameters = calculator_parameters
        self.structure_keyword = structure_keyword
        self.dynamic_structure = DynamicStructure(structure)

    def set_nested(self, builder, path, value):
        """Set a value at a nested attribute or dict path in a process builder.

        Args:
            builder: AiiDA process builder object.
            path: Tuple of keys/attributes leading to the target.
            value: Value to set at the target location.

        Raises:
            AttributeError: If the path cannot be resolved.
        """
        obj = builder
        for key in path[:-1]:
            if hasattr(obj, key):
                obj = getattr(obj, key)
            elif isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                raise AttributeError(f"Cannot find '{key}' in path {path}")

        last_key = path[-1]
        if hasattr(obj, last_key):
            setattr(obj, last_key, value)
        elif isinstance(obj, dict):
            obj[last_key] = value
        else:
            raise AttributeError(f"Cannot set '{last_key}' in path {path}")

    def get_builder(self, x):
        """Return a process builder for the calculator workchain with a structure derived from ``x``.

        Args:
            x: Parameter vector for the distorted structure.

        Returns:
            A process builder with the new structure and calculator parameters set.
        """
        new_ase_structure = self.dynamic_structure(x)
        new_structure = StructureData(ase=new_ase_structure)
        builder = self.calculator.get_builder()
        for key, value in self.parameters.items():
            setattr(builder, key, value)
        self.set_nested(builder, self.structure_keyword, new_structure)
        return builder
