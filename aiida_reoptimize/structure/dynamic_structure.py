from typing import Type

import ase
from aiida.engine import WorkChain
from aiida.orm import Dict
from ase.data import chemical_symbols
from ase.lattice import UnconventionalLattice

from aiida_reoptimize.structure.magmoms_utils import (
    MagneticMomentPreservationError,
    ase_to_structure_preserving_cell_and_magmoms,
)


class ParameterVectorMismatchError(ValueError):
    """Raised when a target vector does not match the lattice parameterization."""


class StructureStandardizationError(ValueError):
    """Raised when a generated structure cannot be standardized with spglib."""


def _plain_dict(value):
    if value is None:
        return {}
    if hasattr(value, "get_dict"):
        return value.get_dict()
    return dict(value)


def _atom_block_symbol(block: dict) -> str | None:
    if "element" in block:
        return block["element"]
    if "z" in block:
        try:
            return chemical_symbols[int(block["z"])]
        except (ValueError, TypeError, IndexError):
            return None
    return None


def _same_atom_id(left, right) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return left == right


def _unique_atom_key(parameters: dict, preferred: str) -> str:
    if preferred not in parameters:
        return preferred

    index = 1
    while f"{preferred}_{index}" in parameters:
        index += 1
    return f"{preferred}_{index}"


def _atom_blocks(parameters: dict):
    for key, value in parameters.items():
        if "atom" in key and isinstance(value, dict):
            yield key, value


def _base_atom_blocks_by_symbol(parameters: dict) -> dict:
    base_atoms_by_symbol = {}
    for _, value in _atom_blocks(parameters):
        symbol = _atom_block_symbol(value)
        if symbol is not None and symbol not in base_atoms_by_symbol:
            base_atoms_by_symbol[symbol] = value
    return base_atoms_by_symbol


def _matching_atom_key_by_id(parameters: dict, atom_id) -> str | None:
    for key, value in _atom_blocks(parameters):
        if "id" in value and _same_atom_id(value["id"], atom_id):
            return key
    return None


def _merge_magnetic_calc_parameters(existing, magnetic_updates: dict) -> dict:
    merged = _plain_dict(existing).copy()
    base_atoms_by_symbol = _base_atom_blocks_by_symbol(merged)

    if "comp" in magnetic_updates:
        merged["comp"] = {
            **merged.get("comp", {}),
            **magnetic_updates["comp"],
        }

    for key, atom_block in _atom_blocks(magnetic_updates):
        symbol = _atom_block_symbol(atom_block)
        merged_atom = {
            **base_atoms_by_symbol.get(symbol, {}),
            **atom_block,
        }

        existing_key = _matching_atom_key_by_id(merged, merged_atom.get("id")) or _unique_atom_key(merged, key)
        merged[existing_key] = merged_atom

    return merged


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

    def _lattice_name(self) -> str:
        return getattr(
            self.__structure_lattice,
            "name",
            self.__structure_lattice.__class__.__name__,
        )

    def _parameter_values(self, x) -> dict[str, float]:
        parameters = list(x)
        if len(parameters) != len(self.__parameter_names):
            expected = ", ".join(self.__parameter_names)
            raise ParameterVectorMismatchError(
                f"Parameter vector length mismatch for {self._lattice_name()}: "
                f"expected {len(self.__parameter_names)} values ({expected}), "
                f"got {len(parameters)}."
            )
        return dict(zip(self.__parameter_names, parameters, strict=True))

    def _cell_from_parameters(self, parameter_values: dict[str, float]):
        try:
            return self.__structure_lattice.__class__(**parameter_values).tocell()
        except UnconventionalLattice:
            return self.__structure_lattice._cell(**parameter_values)

    def __call__(self, x):
        """Create a new ASE Atoms object with cell parameters given by ``x``.

        Args:
            x: Parameter vector matching the free parameters of the Bravais lattice.

        Returns:
            A new ASE Atoms object with the updated cell and scaled positions.
        """
        parameter_values = self._parameter_values(x)
        new_cell = self._cell_from_parameters(parameter_values)
        new_structure = self.__structure.copy()
        new_structure.set_cell(new_cell, scale_atoms=True)
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
        new_structure, magnetic_calc_parameters, _ = ase_to_structure_preserving_cell_and_magmoms(new_ase_structure)
        builder = self.calculator.get_builder()
        parameters = dict(self.parameters)
        if magnetic_calc_parameters:
            if not hasattr(builder, "calc_parameters"):
                raise MagneticMomentPreservationError(
                    "Generated structure has initial magnetic moments, but the calculator builder "
                    "does not accept 'calc_parameters'."
                )
            parameters["calc_parameters"] = Dict(
                dict=_merge_magnetic_calc_parameters(
                    parameters.get("calc_parameters"),
                    magnetic_calc_parameters,
                )
            )

        for key, value in parameters.items():
            setattr(builder, key, value)
        self.set_nested(builder, self.structure_keyword, new_structure)
        return builder
