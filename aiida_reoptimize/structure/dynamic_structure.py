from typing import Type

import ase
from aiida.engine import WorkChain
from aiida.orm import StructureData


class DynamicStructure:
    """
    Class to generate a new structure from a given one by
    changing the cell parameters.
    """

    def __init__(self, structure):
        self.__structure = structure
        self.__structure_lattice = structure.cell.get_bravais_lattice()

    def __call__(self, x):
        new_cell = self.__structure_lattice.__class__(*x)
        new_structure = self.__structure.copy()
        new_structure.set_cell(new_cell.tocell(), scale_atoms=True)
        return new_structure


class StructureCalculator:
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
        """
        Return a process builder for the workchain with
        a new structure generated from x.
        Supports nested attribute/dict paths for structure.
        """
        new_ase_structure = self.dynamic_structure(x)
        new_structure = StructureData(ase=new_ase_structure)
        builder = self.calculator.get_builder()
        for key, value in self.parameters.items():
            setattr(builder, key, value)
        self.set_nested(builder, self.structure_keyword, new_structure)
        return builder