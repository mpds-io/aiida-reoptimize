from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np


def _as_float_array(values: Sequence[Any] | None, *, field_name: str) -> np.ndarray | None:
    if values is None:
        return None

    if isinstance(values, (str, bytes)):
        raise ValueError(f"'{field_name}' must be a sequence of numbers.")

    try:
        array = np.array([float(value) for value in values], dtype=np.float64)
    except TypeError as exc:
        raise ValueError(f"'{field_name}' must be a sequence of numbers.") from exc
    except ValueError as exc:
        raise ValueError(f"'{field_name}' contains a non-numeric value.") from exc

    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"'{field_name}' must be a non-empty 1D sequence.")

    return array


def _extract_initial_parameters_from_structure(structure: Any) -> np.ndarray:
    ase_structure = structure.get_ase() if hasattr(structure, "get_ase") else structure
    lattice = ase_structure.cell.get_bravais_lattice()

    if hasattr(lattice, "vars"):
        values = lattice.vars()
        if isinstance(values, Mapping) and values:
            return np.array([float(value) for value in values.values()], dtype=np.float64)

    raise ValueError("Could not infer structural parameters from structure bravais lattice.")


def _normalize_bounds_from_scalar(scale: float, parameters: np.ndarray) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scalar 'bounds' must be a positive finite number.")

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
        raise ValueError("'bounds' list contains a non-numeric value.") from exc
    except TypeError as exc:
        raise ValueError("'bounds' must be either a number or a list of [low, high] pairs.") from exc

    if bounds_array.ndim != 2 or bounds_array.shape[1] != 2:
        raise ValueError("'bounds' list must have shape (n_parameters, 2).")

    if bounds_array.shape[0] != dimensions:
        raise ValueError(
            "Length mismatch: 'bounds' contains "
            f"{bounds_array.shape[0]} entries but {dimensions} parameters were inferred."
        )

    lower = np.minimum(bounds_array[:, 0], bounds_array[:, 1])
    upper = np.maximum(bounds_array[:, 0], bounds_array[:, 1])

    if np.any(np.isclose(lower, upper)):
        raise ValueError("Each parameter bound must define a non-zero interval.")

    return np.column_stack((lower, upper))


def prepare_optimization_parameters(  # noqa: C901
    parameters: dict[str, Any],
    *,
    structure: Any = None,
    require_bounds: bool,
    require_initial_parameters: bool,
) -> dict[str, Any]:
    """Normalize optimizer parameters and infer the number of variables automatically.

    Contract:
    - Number of variables is inferred and must not be provided by users.
    - ``bounds`` can be either a scalar multiplier or a list of [low, high] pairs.
    """

    if "dimensions" in parameters:
        raise ValueError("'dimensions' must not be passed explicitly. It is inferred automatically.")

    initial_parameters = _as_float_array(parameters.get("initial_parameters"), field_name="initial_parameters")

    if initial_parameters is None and structure is not None:
        initial_parameters = _extract_initial_parameters_from_structure(structure)

    bounds_input = parameters.get("bounds")

    if require_initial_parameters and initial_parameters is None:
        raise ValueError("Could not infer initial parameters. Provide 'initial_parameters' or pass a 'structure'.")

    bounds_array = None
    if bounds_input is None:
        if require_bounds:
            raise ValueError("'bounds' is required for this optimizer.")
    elif isinstance(bounds_input, Real) and not isinstance(bounds_input, bool):
        if initial_parameters is None:
            raise ValueError(
                "Scalar 'bounds' requires inferable parameters. Provide 'initial_parameters' or pass a 'structure'."
            )
        bounds_array = _normalize_bounds_from_scalar(float(bounds_input), initial_parameters)
    else:
        dimensions_for_bounds = int(initial_parameters.size) if initial_parameters is not None else len(bounds_input)
        bounds_array = _normalize_bounds_from_list(bounds_input, dimensions_for_bounds)

    if initial_parameters is not None:
        dimensions = int(initial_parameters.size)
    elif bounds_array is not None:
        dimensions = int(bounds_array.shape[0])
    else:
        raise ValueError(
            "Could not infer the number of variables automatically. Provide a 'structure', "
            "'initial_parameters', or list-form 'bounds'."
        )

    if bounds_array is not None and bounds_array.shape[0] != dimensions:
        raise ValueError("Length mismatch between inferred variable count and normalized bounds.")

    return {
        "dimensions": dimensions,
        "initial_parameters": initial_parameters,
        "bounds": bounds_array,
    }
