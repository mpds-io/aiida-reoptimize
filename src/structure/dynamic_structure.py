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


class DynamicStructureWorkChainGenerator:
    def __init__(
        self,
        structure: ase.atoms.Atoms,
        calculator: Type[WorkChain],
        parameters: dict,
        structure_keyword: str = "structure",
    ):
        self.structure = structure
        self.calculator = calculator
        self.parameters = parameters
        self.structure_keyword = structure_keyword
        self.dynamic_structure = DynamicStructure(structure)

    def get_builder(self, x):
        """
        Return a process builder for the workchain with
        a new structure generated from x.
        """
        # Generate new structure using DynamicStructure
        new_ase_structure = self.dynamic_structure(x)
        new_structure = StructureData(ase=new_ase_structure)
        # Prepare builder
        builder = self.calculator.get_builder()
        for key, value in self.parameters.items():
            setattr(builder, key, value)
        setattr(builder, self.structure_keyword, new_structure)
        return builder
