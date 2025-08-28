
import re
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from aiida.orm import StructureData
from ase import Atoms
from ase.data import chemical_symbols
from spglib import (
    find_primitive,
    get_magnetic_symmetry_dataset,
    standardize_cell,
)


def convert_to_set(data: List[Tuple[Any]]) -> Tuple[set, List[Tuple]]:
    """
    Converts a list of tuples containing mixed data types to a set.

    Handles conversion of numpy arrays and scalars to Python-native types.
    Preserves the structure while ensuring hashability for set operations.

    Args:
        data: List of tuples containing mixed data types

    Returns:
        Tuple containing:
        - A set of unique converted tuples
        - The list of converted tuples (preserving original order)
    """
    # Handle empty input case
    if not data:
        return set(), []

    # Process each tuple element based on its type
    converted_data = []
    for item in data:
        converted_item = []
        for element in item:
            # Convert numpy arrays to tuples for hashability
            if isinstance(element, np.ndarray):
                converted_item.append(tuple(element.tolist()))
            # Convert numpy scalars to their native Python types
            elif isinstance(element, (np.int32, np.float64)):
                converted_item.append(element.item())
            # Preserve other types unchanged
            else:
                converted_item.append(element)
        converted_data.append(tuple(converted_item))

    # Create set from converted data while preserving original list
    return set(converted_data), converted_data


def check_magmoms_ase(atoms: Atoms) -> bool:
    """
    Checks if any atom in the ASE Atoms object has a non-zero magnetic moment.
    If magmoms are lists/arrays (non-collinear case), checks if any element is non-zero.

    Args:
        atoms: ASE Atoms object

    Returns:
        True if any atom has a non-zero magnetic moment, False otherwise.
    """
    return any(
        any(atom.magmom) if hasattr(atom.magmom, "__iter__") else atom.magmom
        for atom in atoms
    )


def numpy_to_python(value: Union[np.ndarray, float]) -> Union[List, float]:
    """
    Converts numpy arrays or scalars to native Python types.

    Args:
        value: Either a numpy array or scalar value

    Returns:
        Native Python float representation of the input value
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.int32, np.float64)):
        return float(value)
    else:
        return value  # Supposed to be already a Python float or list


def convert_ase_to_spg(
    atoms: Atoms,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Converts ASE Atoms object to standardized SPGLIB cell format.

    Args:
        atoms: ASE Atoms object containing structural information

    Returns:
        Tuple containing:
        - lattice vectors (cell matrix)
        - scaled atomic positions
        - atomic numbers
        Optionally magnetic moments if present
    """
    scaled_positions = atoms.get_scaled_positions()
    lattice = atoms.get_cell()[:]
    numbers = atoms.get_atomic_numbers()

    # Check for magnetic moment presence
    if check_magmoms_ase(atoms):
        magmoms = atoms.get_initial_magnetic_moments()
        cell = (lattice, scaled_positions, numbers, magmoms)
    else:
        cell = (lattice, scaled_positions, numbers)
    return cell


def spg_magnetism_handling(cell: Tuple[np.ndarray, ...], return_raw=False):
    """
    Handles magnetic symmetry for space group determination by treating magnetic moments
    as additional atomic properties. This function standardizes the input cell, maps
    unique combinations of atomic types and magnetic tensors, and finds the primitive cell
    of the resulting structure. It returns the primitive cell with restored atomic types
    and magnetic moments. Optionally, it can also return the mapping dictionary used.

    Args:
        cell (Tuple[np.ndarray, ...]): The standardized cell tuple, which must include magnetic moments.
        return_raw (bool): If True, returns additional mapping information for debugging.

    Returns:
        If return_raw is False:
            Tuple containing:
                - Primitive cell lattice vectors (np.ndarray)
                - Primitive cell scaled positions (np.ndarray)
                - List of atomic types (List[int])
                - List of magnetic moments (List)
        If return_raw is True:
            Tuple containing:
                - Primitive cell lattice vectors (np.ndarray)
                - Primitive cell scaled positions (np.ndarray)
                - List of atomic types (List[int])
                - List of mapped atomic types (List[int])
                - List of magnetic moments (List)
            And the mapping dictionary (Dict)
    """
    # This code only works if the cell is standardized.
    # If you provide a primitive cell, it will break because
    # mag_data_set.equivalent_atoms has as many elements as the INPUT cell.
    # For example, if the standardized cell has 4 atoms and the primitive cell has 1,
    # len(mag_data_set.equivalent_atoms) == 1, but len(mag_data_set.std_types) == 4.
    std_cell = spg_get_std(cell)
    mag_data_set = get_magnetic_symmetry_dataset(std_cell)
    data_for_mapping = []

    # Create mapping data for unique types/tensors
    for i in zip(
        mag_data_set.std_types,
        mag_data_set.std_tensors,
        mag_data_set.equivalent_atoms,
        strict=True
    ):
        data_for_mapping.append(i)

    # Convert to set and create mapping
    set_for_maping, converted_data = convert_to_set(data_for_mapping)
    mapper = {val: num for num, val in enumerate(set_for_maping, 1)}

    # Apply mapping to data
    new_types = []
    for i in converted_data:
        new_types.append(mapper[i])

    # Create new cell with mapped types
    new_cell = (
        mag_data_set.std_lattice,
        mag_data_set.std_positions,
        new_types,
    )
    new_prim_cell_ = find_primitive(new_cell)

    # Restore original values
    reverse_mapper = {v: k for k, v in mapper.items()}
    restored_values_ = [reverse_mapper[int(num)] for num in new_prim_cell_[-1]]

    # Separate types and magnetic moments
    restored_types = []
    restored_magmoms = []
    for atoms_prop in restored_values_:
        restored_types.append(atoms_prop[0])
        restored_magmoms.append(atoms_prop[1])

    if return_raw:
        return (
            new_prim_cell_[0],                         # lattice
            new_prim_cell_[1],                         # scaled positions
            restored_types,                            # real atomic types
            new_prim_cell_[2],                         # mapped atomic types
            restored_magmoms,                          # real magnetic moments
        ), {str(v): k for k, v in mapper.items()}      # mapping dictionary

    return (
        new_prim_cell_[0],  # lattice
        new_prim_cell_[1],  # scaled positions
        restored_types,     # real atomic types
        restored_magmoms,   # real magnetic moments
    )


def spg_get_primitive(
    cell: Tuple[np.ndarray, ...],
) -> Union[
    Tuple[np.ndarray, np.ndarray, List[int]],
    Tuple[np.ndarray, np.ndarray, List[int], List[float]],
]:
    """
    Determines primitive cell representation based on magnetic properties.

    Args:
        cell: Cell tuple containing structural information

    Returns:
        Primitive cell representation (with optional magnetic moments)
    """
    if len(cell) == 3:
        return find_primitive(cell)
    return spg_magnetism_handling(cell)


def ase_to_prim(atoms: Atoms) -> Atoms:
    """
    Converts an ASE Atoms object to its primitive cell representation,
    including magnetic moments if present.

    Args:
        atoms: ASE Atoms object

    Returns:
        ASE Atoms object in primitive cell form
    """
    struct_raw = convert_ase_to_spg(atoms)
    struct_prim = spg_get_primitive(struct_raw)

    # Create atoms object with magnetic moments if present
    if check_magmoms_ase(atoms):
        new_atoms = Atoms(
            cell=struct_prim[0],
            scaled_positions=struct_prim[1],
            numbers=struct_prim[2],
            magmoms=struct_prim[3],
            pbc=atoms.pbc,
        )
    else:
        new_atoms = Atoms(
            cell=struct_prim[0],
            scaled_positions=struct_prim[1],
            numbers=struct_prim[2],
            pbc=atoms.pbc,
        )
    return new_atoms


def spg_get_std(cell):
    """
    Returns the standardized cell using SPGLIB, including magnetic tensors if present.

    Args:
        cell: Cell tuple (with or without magnetic moments)

    Returns:
        Standardized cell tuple
    """
    if len(cell) == 3:
        return standardize_cell(cell)
    msg_dataset = get_magnetic_symmetry_dataset(cell)
    return (
        msg_dataset.std_lattice,
        msg_dataset.std_positions,
        msg_dataset.std_types,
        msg_dataset.std_tensors,
    )


def ase_to_std(atoms: Atoms) -> Atoms:
    """
    Converts an ASE Atoms object to its standardized cell representation,
    including magnetic moments if present.

    Args:
        atoms: ASE Atoms object

    Returns:
        ASE Atoms object in standardized cell form
    """
    struct_raw = convert_ase_to_spg(atoms)
    struct_prim = spg_get_std(struct_raw)

    # Create atoms object with magnetic moments if present
    if check_magmoms_ase(atoms):
        new_atoms = Atoms(
            cell=struct_prim[0],
            scaled_positions=struct_prim[1],
            numbers=struct_prim[2],
            magmoms=struct_prim[3],
            pbc=atoms.pbc,
        )
    else:
        new_atoms = Atoms(
            cell=struct_prim[0],
            scaled_positions=struct_prim[1],
            numbers=struct_prim[2],
            pbc=atoms.pbc,
        )
    return new_atoms


def numbers_to_symbols(numbers: List[int]) -> List[str]:
    """
    Converts atomic numbers to chemical symbols.

    Args:
        numbers: List of atomic numbers

    Returns:
        List of chemical symbols
    """
    return [
        chemical_symbols[num]
        for num in numbers
        if 1 <= num < len(chemical_symbols)
    ]


def ase_to_struct_prim(atoms: Atoms) -> Tuple[StructureData, Dict[int, Tuple]]:
    """
    Converts an ASE Atoms object to an AiiDA StructureData object in primitive cell form,
    including magnetic moments and kind mapping if present.

    Args:
        atoms: ASE Atoms object

    Returns:
        Tuple of (StructureData object, kind mapping dictionary)
    """
    # It expects to get primitive cell as input
    ase_prim = ase_to_prim(atoms)

    if check_magmoms_ase(atoms):
        cell_raw = convert_ase_to_spg(atoms)
        # Expects tuple of length 5 and dict
        # (lattice, scaled_positions, numbers, kinds, magmoms), mapper
        cell_with_kinds, mapper = spg_magnetism_handling(
            cell_raw, return_raw=True
        )
        atom_symbols = numbers_to_symbols(cell_with_kinds[2])
        atom_kinds = [
            f"{sym}{num}" for sym, num in zip(atom_symbols, cell_with_kinds[3], strict=True)
        ]
        # XXX Should atomic position values cropped?
        # Tests shows that it's fine without cropping, but if
        # numerical error will be bigger than 1e-5, it can cause problems
        # mpmath package can do this properly if needed
        atom_positions = ase_prim.get_positions()
        structure = StructureData(cell=cell_with_kinds[0])
        for pos, sym, kind in zip(atom_positions, atom_symbols, atom_kinds, strict=True):
            structure.append_atom(position=pos, symbols=sym, name=kind)
    else:
        mapper = dict()
        structure = StructureData(ase=ase_prim)
    return structure, mapper


def reverse_structure_data(
    structure: StructureData, mapper: Dict[int, Tuple], pbc=None
) -> Atoms:
    """
    Flexible version with customizable periodic boundary conditions.

    Args:
        structure: StructureData object with atoms
        mapper: Dictionary of the form {number: (atomic_number, magnetic_moment, orbital)}
        pbc: Periodic boundary conditions (defaults to those in structure)

    Returns:
        ASE Atoms object with magnetic moments set
    """
    # Get data from StructureData
    sites = structure.sites
    cell = structure.cell

    # Determine PBC
    if pbc is None:
        # Try to get from structure, otherwise use True for all axes
        try:
            pbc = structure.pbc
        except AttributeError:
            pbc = [True, True, True]

    positions = []
    symbols = []
    magnetic_moments = []

    # Extract positions, symbols, and magnetic moments
    for site in sites:
        positions.append(site.position)

        # Get atomic symbol from kind name
        kind_name = site.kind_name
        symbol_match = re.match(r"([A-Za-z]+)\d*$", kind_name)
        if symbol_match:
            symbol = symbol_match.group(1)
            symbols.append(symbol)
        else:
            raise ValueError(
                f"Cannot extract symbol from kind name: {kind_name}"
            )

        # Extract magnetic moment from kind name
        match = re.search(r"[A-Za-z]+(\d+)$", kind_name)
        if match:
            atom_number = match.group(1)

            if atom_number in mapper:
                _, magnetic_moment, _ = mapper[atom_number]
                magnetic_moments.append(magnetic_moment)
            else:
                magnetic_moments.append(0.0)
        else:
            raise ValueError(
                f"Cannot extract number from kind name: {kind_name}"
            )

    # Create ASE Atoms object
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)

    # Set magnetic moments
    atoms.set_initial_magnetic_moments(magnetic_moments)

    return atoms