from typing import Type

from aiida.common.exceptions import NotExistent
from aiida.orm import load_node


class BasicExtractor:
    def __init__(self, node_exctractor: Type[callable], penalty=1e+10):
        self.node_extractor = node_exctractor
        self.penalty = penalty

    def __call__(self, results: list):
        values = []
        for item in results:
            if item["status"] == "ok":
                try:
                    node = load_node(item["pk"])
                    value = self.node_extractor(node.outputs)
                except NotExistent:
                    value = self.penalty
            else:
                value = self.penalty
            values.append(value)
        return values