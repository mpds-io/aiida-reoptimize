"""Result extractors for evaluator workchains."""

from typing import Any, Callable

from aiida.common.exceptions import NotExistent
from aiida.orm import load_node


class BasicExtractor:
    """Extract scalar values from finished AiiDA processes with fallback penalty."""

    def __init__(
        self,
        node_extractor: Callable[[Any], Any],
        penalty: float = 1e10,
    ):
        self.node_extractor = node_extractor
        self.penalty = penalty

    def __call__(self, results: list[dict[str, Any]]) -> list[Any]:
        values = []
        for item in results:
            value = self.penalty
            if item.get("status") == "ok":
                try:
                    node = load_node(item.get("pk"))
                    extracted = self.node_extractor(node.outputs)
                    if extracted is not None:
                        # Sometimes Aida returns Python types,
                        # This check allows you to catch it
                        value = extracted.value if hasattr(extracted, "value") else extracted
                except NotExistent:
                    pass
            values.append(value)
        return values

    def get_penalty(self):
        """Return the penalty value."""
        return self.penalty
