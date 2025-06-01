from aiida.orm import QueryBuilder
from aiida.orm.nodes.data.code import Code


def find_nodes(*args):
    qb = QueryBuilder()
    qb.append(Code, filters={"label": {"in": [i for i in args]}})  # noqa: E501
    nodes = qb.all()
    
    if not nodes:
        raise ValueError("No Fleur or inpgen codes found in the database")
    
    data = {}
    for node in nodes:
        data[node[0].label] = node[0].pk
    return data