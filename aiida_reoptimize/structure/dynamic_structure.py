import inspect
import logging
from collections.abc import Callable
from typing import Any, Type

import ase
from aiida.engine import WorkChain
from aiida.orm import Dict
from ase.data import chemical_symbols

from aiida_reoptimize.structure.magmoms_utils import (
    MagneticMomentPreservationError,
    ase_to_structure_preserving_cell_and_magmoms,
)

LOGGER = logging.getLogger(__name__)


class ParameterVectorMismatchError(ValueError):
    """Raised when a target vector does not match the lattice parameterization."""


class StructureGenerationError(ValueError):
    """Raised when a generated structure or builder cannot be assembled."""


class StructureStandardizationError(ValueError):
    """Raised when a generated structure cannot be standardized with spglib."""


def _python_value(value: Any) -> Any:
    """Return logging-friendly Python-native values for numpy/ASE objects."""

    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _object_name(value: Any) -> str:
    cls = value if inspect.isclass(value) else value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _signature(value: Any) -> str:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return list(value)


def _format_mapping(mapping: dict[str, Any]) -> str:
    if not mapping:
        return "{}"
    return ", ".join(f"{key}={value!r}" for key, value in mapping.items())


def _format_lines(title: str, values: dict[str, Any]) -> str:
    lines = [title]
    for key, value in values.items():
        rendered = _format_mapping(value) if isinstance(value, dict) else repr(value)
        lines.append(f"  {key}: {rendered}")
    return "\n".join(lines)


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

    def _initial_parameter_values(self) -> dict[str, Any]:
        values = self.__structure_lattice.vars()
        return {name: _python_value(values[name]) for name in self.__parameter_names}

    def _candidate_preview(self, x) -> tuple[list[Any], dict[str, Any]]:
        parameters = [_python_value(value) for value in _as_list(x)]
        labels = list(self.__parameter_names)
        if len(parameters) > len(labels):
            labels.extend(f"<extra_{index}>" for index in range(len(labels), len(parameters)))
        preview = {
            labels[index] if index < len(labels) else f"<value_{index}>": value
            for index, value in enumerate(parameters)
        }
        return parameters, preview

    def _reference_summary(self) -> dict[str, Any]:
        return {
            "formula": self.__structure.get_chemical_formula(),
            "natoms": len(self.__structure),
            "pbc": tuple(bool(value) for value in self.__structure.pbc),
            "cell": _python_value(self.__structure.cell.array),
        }

    def diagnostics(self, x=None, stage: str | None = None, target_index: int | None = None) -> dict[str, Any]:
        """Return structured diagnostics for the lattice-to-cell generation boundary."""

        details: dict[str, Any] = {
            "stage": stage or "dynamic_structure",
            "target_index": target_index,
            "lattice_name": self._lattice_name(),
            "lattice_class": _object_name(self.__structure_lattice),
            "lattice_constructor_signature": _signature(self.__structure_lattice.__class__),
            "cell_builder_signature": _signature(self.__structure_lattice._cell),
            "expected_parameter_names": self.__parameter_names,
            "expected_parameter_count": len(self.__parameter_names),
            "initial_parameters": self._initial_parameter_values(),
            "reference_structure": self._reference_summary(),
        }
        if x is not None:
            try:
                candidate_values, candidate_preview = self._candidate_preview(x)
            except TypeError:
                details.update(
                    {
                        "candidate_type": _object_name(x),
                        "candidate_error": "candidate target is not iterable",
                    }
                )
            else:
                missing = self.__parameter_names[len(candidate_values) :]
                details.update(
                    {
                        "candidate_type": _object_name(x),
                        "candidate_length": len(candidate_values),
                        "candidate_values": candidate_values,
                        "candidate_parameter_preview": candidate_preview,
                        "missing_parameters": missing,
                        "extra_value_count": max(0, len(candidate_values) - len(self.__parameter_names)),
                    }
                )
        return details

    def format_diagnostics(self, x=None, stage: str | None = None, target_index: int | None = None) -> str:
        """Return a compact multi-line diagnostic report for AiiDA workchain logs."""

        details = self.diagnostics(x=x, stage=stage, target_index=target_index)
        return _format_lines("Structure generation diagnostics:", details)

    def _parameter_values(self, x) -> dict[str, float]:
        try:
            parameters = _as_list(x)
        except TypeError as exc:
            raise ParameterVectorMismatchError(
                f"Parameter vector for {self._lattice_name()} must be iterable; got {_object_name(x)}.\n"
                f"{self.format_diagnostics(x=x, stage='parameter_validation')}"
            ) from exc
        if len(parameters) != len(self.__parameter_names):
            expected = ", ".join(self.__parameter_names)
            raise ParameterVectorMismatchError(
                f"Parameter vector length mismatch for {self._lattice_name()}: "
                f"expected {len(self.__parameter_names)} values ({expected}), "
                f"got {len(parameters)}.\n"
                f"{self.format_diagnostics(x=parameters, stage='parameter_validation')}"
            )
        return dict(zip(self.__parameter_names, parameters, strict=True))

    def _cell_from_parameters(self, parameter_values: dict[str, float]):
        return self.__structure_lattice._cell(**parameter_values)

    def __call__(self, x):
        """Create a new ASE Atoms object with cell parameters given by ``x``.

        Args:
            x: Parameter vector matching the free parameters of the Bravais lattice.

        Returns:
            A new ASE Atoms object with the updated cell and scaled positions.
        """
        parameter_values = self._parameter_values(x)
        LOGGER.debug("%s", self.format_diagnostics(x=x, stage="cell_generation"))
        try:
            new_cell = self._cell_from_parameters(parameter_values)
        except Exception as exc:
            diagnostics = self.format_diagnostics(x=x, stage="cell_generation")
            LOGGER.exception("Failed to build a cell from lattice parameters.\n%s", diagnostics)
            raise StructureGenerationError(
                f"Failed to build a cell from lattice parameters: {exc}\n{diagnostics}"
            ) from exc
        new_structure = self.__structure.copy()
        new_structure.set_cell(new_cell, scale_atoms=True)
        LOGGER.debug("Generated ASE structure cell for %s: %r", self._lattice_name(), _python_value(new_cell))
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
        reporter: Callable[[str], None] | None = None,
    ):
        self.structure = structure
        self.calculator = calculator
        self.parameters = calculator_parameters
        self.structure_keyword = structure_keyword
        self.dynamic_structure = DynamicStructure(structure)
        self.reporter = reporter

    def _report(self, message: str) -> None:
        LOGGER.info("%s", message)
        if self.reporter is not None:
            self.reporter(message)

    def format_diagnostics(self, x=None, stage: str | None = None, target_index: int | None = None) -> str:
        """Return diagnostics that include both lattice and builder context."""

        dynamic_details = self.dynamic_structure.diagnostics(x=x, stage=stage, target_index=target_index)
        dynamic_details.update(
            {
                "calculator_workchain": _object_name(self.calculator),
                "structure_keyword": self.structure_keyword,
                "calculator_parameter_keys": tuple(sorted(self.parameters.keys())),
            }
        )
        return _format_lines("Structure calculator diagnostics:", dynamic_details)

    def set_nested(self, builder, path, value):
        """Set a value at a nested attribute or dict path in a process builder.

        Args:
            builder: AiiDA process builder object.
            path: Tuple of keys/attributes leading to the target.
            value: Value to set at the target location.

        Raises:
            StructureGenerationError: If the path cannot be resolved.
        """
        obj = builder
        for key in path[:-1]:
            if hasattr(obj, key):
                obj = getattr(obj, key)
            elif isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                raise StructureGenerationError(f"Cannot find '{key}' in structure input path {path}")

        last_key = path[-1]
        if hasattr(obj, last_key):
            setattr(obj, last_key, value)
        elif isinstance(obj, dict):
            obj[last_key] = value
        else:
            raise StructureGenerationError(f"Cannot set '{last_key}' in structure input path {path}")

    def get_builder(self, x, target_index: int | None = None):
        """Return a process builder for the calculator workchain with a structure derived from ``x``.

        Args:
            x: Parameter vector for the distorted structure.
            target_index: Optional index of ``x`` in the submitted target batch.

        Returns:
            A process builder with the new structure and calculator parameters set.
        """
        try:
            new_ase_structure = self.dynamic_structure(x)
        except ParameterVectorMismatchError:
            diagnostics = self.format_diagnostics(x=x, stage="parameter_validation", target_index=target_index)
            self._report(diagnostics)
            raise
        except StructureGenerationError:
            diagnostics = self.format_diagnostics(x=x, stage="dynamic_structure", target_index=target_index)
            self._report(diagnostics)
            raise

        try:
            new_structure, magnetic_calc_parameters, _ = ase_to_structure_preserving_cell_and_magmoms(new_ase_structure)
        except MagneticMomentPreservationError:
            diagnostics = self.format_diagnostics(x=x, stage="magnetic_moment_conversion", target_index=target_index)
            self._report(diagnostics)
            raise
        except Exception as exc:
            diagnostics = self.format_diagnostics(x=x, stage="aiida_structure_conversion", target_index=target_index)
            LOGGER.exception("Failed to convert generated ASE structure to AiiDA StructureData.\n%s", diagnostics)
            raise StructureGenerationError(
                f"Failed to convert generated ASE structure to AiiDA StructureData: {exc}\n{diagnostics}"
            ) from exc

        try:
            builder = self.calculator.get_builder()
        except Exception as exc:
            diagnostics = self.format_diagnostics(x=x, stage="builder_creation", target_index=target_index)
            LOGGER.exception("Failed to create calculator builder.\n%s", diagnostics)
            raise StructureGenerationError(f"Failed to create calculator builder: {exc}\n{diagnostics}") from exc

        parameters = dict(self.parameters)
        if magnetic_calc_parameters:
            if not hasattr(builder, "calc_parameters"):
                diagnostics = self.format_diagnostics(
                    x=x, stage="magnetic_parameter_injection", target_index=target_index
                )
                self._report(diagnostics)
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
        try:
            self.set_nested(builder, self.structure_keyword, new_structure)
        except StructureGenerationError as exc:
            diagnostics = self.format_diagnostics(x=x, stage="structure_input_injection", target_index=target_index)
            LOGGER.exception("Failed to inject generated structure into calculator builder.\n%s", diagnostics)
            raise StructureGenerationError(f"{exc}\n{diagnostics}") from exc
        return builder
