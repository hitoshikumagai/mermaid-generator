from importlib import import_module
from typing import Any

__all__ = [
    "build_mock_graph",
    "calculate_layout_positions",
    "export_to_mermaid",
    "to_flow_node_specs",
    "to_flow_edge_specs",
    "flow_items_to_graph_data",
]


def __getattr__(name: str) -> Any:
    if name in {"build_mock_graph", "calculate_layout_positions", "export_to_mermaid"}:
        module = import_module(".graph_logic", __name__)
        return getattr(module, name)
    if name in {"to_flow_node_specs", "to_flow_edge_specs", "flow_items_to_graph_data"}:
        module = import_module(".ui_mapper", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
