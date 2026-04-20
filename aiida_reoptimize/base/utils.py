from aiida.orm.nodes.data.code import Code
from aiida.orm.querybuilder import QueryBuilder


def find_nodes(*args):
    """Query the AiiDA database for Code nodes by label.

    Args:
        *args: Code label strings to search for.

    Returns:
        Dictionary mapping code labels to their PKs.

    Raises:
        ValueError: If no matching codes are found.
    """
    qb = QueryBuilder()
    qb.append(Code, filters={"label": {"in": [i for i in args]}})
    nodes = qb.all()

    if not nodes:
        raise ValueError("No Fleur or inpgen codes found in the database")

    data = {}
    for node in nodes:
        data[node[0].label] = node[0].pk
    return data
