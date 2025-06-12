import os

import numpy as np
from mpds_client import MPDSDataRetrieval


def get_geometry_MPDS(mpds_query: dict):
    # from https://github.com/mpds-io/mpds-aiida/blob/master/mpds_aiida/workflows/mpds.py
    """
    Getting geometry from MPDS database.
    """
    # check for API key
    api_key = os.getenv("MPDS_KEY")
    if not api_key:
        raise Exception(
            "MPDS API key not found. Please set the MPDS_KEY environment variable."  # noqa: E501
        )
    client = MPDSDataRetrieval(api_key=api_key)

    query_dict = mpds_query
    # prepare query
    query_dict["props"] = "atomic structure"
    if "classes" in query_dict:
        query_dict["classes"] += ", non-disordered"
    else:
        query_dict["classes"] = "non-disordered"

    try:
        answer = client.get_data(
            query_dict,
            fields={"S": ["cell_abc", "sg_n", "basis_noneq", "els_noneq"]},
        )
    except Exception as ex:
        raise Exception(f"Error retrieving data from MPDS: {ex}") from ex

    structs = [client.compile_crystal(line, flavor="ase") for line in answer]
    structs = list(filter(None, structs))

    if not structs:
        raise Exception("No structures found in MPDS database.")

    minimal_struct = min([len(s) for s in structs])

    # get structures with minimal number
    # of atoms and find the one with median cell vectors
    cells = np.array([
        s.get_cell().reshape(9) for s in structs if len(s) == minimal_struct
    ])
    median_cell = np.median(cells, axis=0)
    median_idx = int(
        np.argmin(np.sum((cells - median_cell) ** 2, axis=1) ** 0.5)
    )
    return structs[median_idx]


if __name__ == "__main__":
    phase = ("WS2", 194)
    formula, sgs = phase
    structure = get_geometry_MPDS({'formulae': formula, 'sgs': sgs})
    print(structure)

