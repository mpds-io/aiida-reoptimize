from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np


class OptimizationParameterError(ValueError):
    """Raised when optimizer input parameters cannot be normalized."""


def _python_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _format_mapping(mapping: Mapping[str, Any]) -> str:
    if not mapping:
        return "{}"
    return ", ".join(f"{key}={value!r}" for key, value in mapping.items())


def _format_lines(title: str, values: Mapping[str, Any]) -> str:
    lines = [title]
    for key, value in values.items():
        rendered = _format_mapping(value) if isinstance(value, Mapping) else repr(value)
        lines.append(f"  {key}: {rendered}")
    return "\n".join(lines)


def _candidate_preview(values: Sequence[Any], parameter_names: tuple[str, ...]) -> dict[str, Any]:
    candidate_values = [_python_value(value) for value in values]
    labels = list(parameter_names)
    if len(candidate_values) > len(labels):
        labels.extend(f"<extra_{index}>" for index in range(len(labels), len(candidate_values)))
    return {
        labels[index] if index < len(labels) else f"<value_{index}>": value
        for index, value in enumerate(candidate_values)
    }


def _structure_lattice_context(structure: Any) -> dict[str, Any]:
    ase_structure = structure.get_ase() if hasattr(structure, "get_ase") else structure
    lattice = ase_structure.cell.get_bravais_lattice()

    if not hasattr(lattice, "vars"):
        raise OptimizationParameterError("Could not infer structural parameters from structure bravais lattice.")

    values = lattice.vars()
    if not isinstance(values, Mapping) or not values:
        raise OptimizationParameterError("Could not infer structural parameters from structure bravais lattice.")

    parameter_names = tuple(getattr(lattice, "parameters", tuple(values)))
    lattice_name = getattr(lattice, "name", lattice.__class__.__name__)
    parameter_values = np.array([float(values[name]) for name in parameter_names], dtype=np.float64)
    return {
        "ase_structure": ase_structure,
        "lattice": lattice,
        "lattice_name": lattice_name,
        "lattice_class": f"{lattice.__class__.__module__}.{lattice.__class__.__qualname__}",
        "parameter_names": parameter_names,
        "parameter_values": parameter_values,
        "cell": _python_value(ase_structure.cell.array),
        "formula": ase_structure.get_chemical_formula(),
    }


def _parameter_diagnostics(
    *,
    field_name: str,
    provided: Sequence[Any] | np.ndarray | None = None,
    structure_context: Mapping[str, Any] | None = None,
    bounds: Any = None,
) -> str:
    details: dict[str, Any] = {"field_name": field_name}
    if structure_context is not None:
        parameter_names = structure_context["parameter_names"]
        details.update(
            {
                "structure_formula": structure_context["formula"],
                "lattice_name": structure_context["lattice_name"],
                "lattice_class": structure_context["lattice_class"],
                "expected_parameter_names": parameter_names,
                "expected_parameter_count": len(parameter_names),
                "structure_initial_parameters": _candidate_preview(
                    structure_context["parameter_values"],
                    parameter_names,
                ),
                "structure_cell": structure_context["cell"],
            }
        )
    else:
        parameter_names = ()

    if provided is not None:
        provided_values = [_python_value(value) for value in provided]
        details.update(
            {
                "provided_count": len(provided_values),
                "provided_values": provided_values,
                "provided_parameter_preview": _candidate_preview(provided_values, parameter_names),
                "extra_value_count": max(0, len(provided_values) - len(parameter_names)) if parameter_names else 0,
                "missing_parameters": parameter_names[len(provided_values) :] if parameter_names else (),
            }
        )

    if bounds is not None:
        details["bounds"] = _python_value(bounds)

    return _format_lines("Optimization parameter diagnostics:", details)


def _as_float_array(values: Sequence[Any] | None, *, field_name: str) -> np.ndarray | None:
    if values is None:
        return None

    if hasattr(values, "get_list"):
        values = values.get_list()

    if isinstance(values, (str, bytes)):
        raise OptimizationParameterError(f"'{field_name}' must be a sequence of numbers.")

    try:
        array = np.array([float(value) for value in values], dtype=np.float64)
    except TypeError as exc:
        raise OptimizationParameterError(f"'{field_name}' must be a sequence of numbers.") from exc
    except ValueError as exc:
        raise OptimizationParameterError(f"'{field_name}' contains a non-numeric value.") from exc

    if array.ndim != 1 or array.size == 0:
        raise OptimizationParameterError(f"'{field_name}' must be a non-empty 1D sequence.")

    return array


def _extract_lattice_parameters_from_structure(structure: Any) -> tuple[np.ndarray, tuple[str, ...], str]:
    context = _structure_lattice_context(structure)
    return context["parameter_values"], context["parameter_names"], context["lattice_name"]


def _extract_initial_parameters_from_structure(structure: Any) -> np.ndarray:
    parameters, _, _ = _extract_lattice_parameters_from_structure(structure)
    return parameters


def _validate_parameters_match_structure(parameters: np.ndarray, structure: Any, *, field_name: str) -> None:
    context = _structure_lattice_context(structure)
    structure_parameters = context["parameter_values"]
    parameter_names = context["parameter_names"]
    lattice_name = context["lattice_name"]
    if parameters.size == structure_parameters.size:
        return

    expected = ", ".join(parameter_names)
    diagnostics = _parameter_diagnostics(
        field_name=field_name,
        provided=parameters,
        structure_context=context,
    )
    raise OptimizationParameterError(
        f"Length mismatch: '{field_name}' contains {parameters.size} entries but "
        f"{lattice_name} expects {structure_parameters.size} parameters ({expected}).\n{diagnostics}"
    )


def _normalize_bounds_from_scalar(scale: float, parameters: np.ndarray) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        raise OptimizationParameterError("Scalar 'bounds' must be a positive finite number.")

    normalized = []
    for parameter in parameters:
        low_scale, high_scale = (1 - scale, 1 + scale)
        low, high = low_scale * parameter, high_scale * parameter
        normalized.append([low, high])

    return np.array(normalized, dtype=np.float64)


def _normalize_bounds_from_list(bounds: Any, dimensions: int) -> np.ndarray:
    try:
        bounds_array = np.array(bounds, dtype=np.float64)
    except ValueError as exc:
        raise OptimizationParameterError("'bounds' list contains a non-numeric value.") from exc
    except TypeError as exc:
        raise OptimizationParameterError("'bounds' must be either a number or a list of [low, high] pairs.") from exc

    if bounds_array.ndim != 2 or bounds_array.shape[1] != 2:
        raise OptimizationParameterError("'bounds' list must have shape (n_parameters, 2).")

    if bounds_array.shape[0] != dimensions:
        raise OptimizationParameterError(
            "Length mismatch: 'bounds' contains "
            f"{bounds_array.shape[0]} entries but {dimensions} parameters were inferred."
        )

    lower = np.minimum(bounds_array[:, 0], bounds_array[:, 1])
    upper = np.maximum(bounds_array[:, 0], bounds_array[:, 1])

    if np.any(np.isclose(lower, upper)):
        raise OptimizationParameterError("Each parameter bound must define a non-zero interval.")

    return np.column_stack((lower, upper))


def prepare_optimization_parameters(  # noqa: C901
    parameters: dict[str, Any],
    *,
    structure: Any = None,
    require_bounds: bool,
    require_initial_parameters: bool,
    reporter: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Normalize optimizer parameters and infer the number of variables automatically.

    Contract:
    - Number of variables is inferred and must not be provided by users.
    - ``bounds`` can be either a scalar multiplier or a list of [low, high] pairs.
    """

    if "dimensions" in parameters:
        raise OptimizationParameterError("'dimensions' must not be passed explicitly. It is inferred automatically.")

    initial_parameters = _as_float_array(parameters.get("initial_parameters"), field_name="initial_parameters")
    structure_context = _structure_lattice_context(structure) if structure is not None else None

    if initial_parameters is not None and structure_context is not None:
        structure_parameters = structure_context["parameter_values"]
        if initial_parameters.size != structure_parameters.size:
            parameter_names = structure_context["parameter_names"]
            expected = ", ".join(parameter_names)
            diagnostics = _parameter_diagnostics(
                field_name="initial_parameters",
                provided=initial_parameters,
                structure_context=structure_context,
            )
            message = (
                "Length mismatch: 'initial_parameters' contains "
                f"{initial_parameters.size} entries but {structure_context['lattice_name']} expects "
                f"{structure_parameters.size} parameters ({expected}). "
                "Using structure-inferred initial_parameters instead."
            )
            if reporter is not None:
                reporter(f"{message}\n{diagnostics}")
            initial_parameters = structure_parameters.copy()

    if initial_parameters is None and structure_context is not None:
        initial_parameters = structure_context["parameter_values"].copy()

    bounds_input = parameters.get("bounds")

    if require_initial_parameters and initial_parameters is None:
        raise OptimizationParameterError(
            "Could not infer initial parameters. Provide 'initial_parameters' or pass a 'structure'."
        )

    bounds_array = None
    if bounds_input is None:
        if require_bounds:
            raise OptimizationParameterError("'bounds' is required for this optimizer.")
    elif isinstance(bounds_input, Real) and not isinstance(bounds_input, bool):
        if initial_parameters is None:
            raise OptimizationParameterError(
                "Scalar 'bounds' requires inferable parameters. Provide 'initial_parameters' or pass a 'structure'."
            )
        bounds_array = _normalize_bounds_from_scalar(float(bounds_input), initial_parameters)
    else:
        dimensions_for_bounds = int(initial_parameters.size) if initial_parameters is not None else len(bounds_input)
        try:
            bounds_array = _normalize_bounds_from_list(bounds_input, dimensions_for_bounds)
        except OptimizationParameterError as exc:
            diagnostics = _parameter_diagnostics(
                field_name="bounds",
                provided=initial_parameters,
                structure_context=structure_context,
                bounds=bounds_input,
            )
            raise OptimizationParameterError(f"{exc}\n{diagnostics}") from exc

    if initial_parameters is not None:
        dimensions = int(initial_parameters.size)
    elif bounds_array is not None:
        dimensions = int(bounds_array.shape[0])
    else:
        raise OptimizationParameterError(
            "Could not infer the number of variables automatically. Provide a 'structure', "
            "'initial_parameters', or list-form 'bounds'."
        )

    if bounds_array is not None and bounds_array.shape[0] != dimensions:
        raise OptimizationParameterError("Length mismatch between inferred variable count and normalized bounds.")

    return {
        "dimensions": dimensions,
        "initial_parameters": initial_parameters,
        "bounds": bounds_array,
    }
