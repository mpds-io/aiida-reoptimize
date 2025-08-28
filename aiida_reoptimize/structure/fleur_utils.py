import os
import subprocess
import tempfile
from io import StringIO

from aiida_fleur.data.fleurinp import FleurinpData
from ase import Atoms
from ase.io import write as ase_write


class Fleur_setup:
    """Class to prepare input for inpgen."""

    def __init__(self, ase_obj):
        self.ase_obj = ase_obj

    def validate(self):
        self.xml_input = self.ase_to_fleur_xml(self.ase_obj)
        if not self.xml_input:
            return "Fleur inpgen misconfiguration occured"
        return None

    def get_input_setup(self, label):
        if self.xml_input:
            return self.xml_input.replace("%ABSDX_%", label)

    def ase_to_fleur_xml(self, ase_obj: Atoms):
        """
        Skipping the textual Fleur input generation
        in order to simplify our provenance persistence layers
        """
        buff = StringIO()
        ase_write(
            buff,
            ase_obj,
            format="fleur-inpgen",
            parameters={
                "title": "%ABSDX_%",
            },
        )
        buff.seek(0)
        txt_input = buff.getvalue()

        with tempfile.TemporaryDirectory(prefix="fleur_inpgen_") as tmp_dir:
            input_path = os.path.join(tmp_dir, "fleur.inp")
            with open(input_path, "w") as f:
                f.write(txt_input)

            opts = ["-f", "fleur.inp", "-inc", "+all", "-noco"]

            inpgen_path = os.environ.get("FLEUR_INPGEN_PATH")
            if not inpgen_path or not os.path.exists(inpgen_path):
                raise FileNotFoundError("FLEUR_INPGEN_PATH is not set or does not exist")

            p = subprocess.Popen(
                [inpgen_path] + opts,
                cwd=tmp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr = p.communicate()
            if p.returncode != 0:
                error_msg = stderr.decode() if stderr else "inpgen failed with unknown error"
                raise RuntimeError(f"inpgen failed: {error_msg}")

            xml_path = os.path.join(tmp_dir, "inp.xml")
            if not os.path.exists(xml_path):
                error_msg = stderr.decode() if stderr else "inpgen produced no result"
                raise RuntimeError(f"inpgen produced no result: {error_msg}")

            with open(xml_path, "r") as f:
                xml_input = f.read()

        return xml_input


def convert_xml_to_FleurInpData(xml_input: str):
    """
    Takes the content of an inp.xml file as a string, saves it to a temporary folder
    as 'inp.xml', and converts it to a FleurinpData object.

    Args:
        xml_input (str): Content of inp.xml as a string.

    Returns:
        FleurinpData: The FleurinpData object created from the XML input.
    """

    # !!! IF YOU WORK WITH MAGMOMS IT IS HIGHLY IMPORTANT TO MAKE SURE THAT
    # !!! YOU ARE USING THIS ase-fleur LIBRARY git+https://github.com/blokhin/ase-fleur

    with tempfile.TemporaryDirectory() as tmp_dir:
        xml_path = os.path.join(tmp_dir, "inp.xml")
        with open(xml_path, "w") as f:
            f.write(xml_input)
        fleur_inp_data = FleurinpData(files=[xml_path])

    return fleur_inp_data
