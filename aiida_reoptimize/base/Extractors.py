from typing import Type

from aiida.common.exceptions import NotExistent
from aiida.orm import load_node


class BasicExtractor:
    def __init__(self, node_extractor: Type[callable], penalty=1e+10):
        self.node_extractor = node_extractor
        self.penalty = penalty

    def __call__(self, results: list):
        values = []
        for item in results:
            value = self.penalty
            if item.get("status") == "ok":
                try:
                    node = load_node(item.get("pk"))
                    extracted = self.node_extractor(node.outputs)
                    if extracted is not None:
                        value = extracted
                except NotExistent:
                    pass
            values.append(value)
        return values
