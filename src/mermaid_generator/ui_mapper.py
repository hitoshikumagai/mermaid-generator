from typing import Any, Dict, List, Mapping, Sequence, Tuple

NodeData = Dict[str, str]
EdgeData = Dict[str, str]
PositionMap = Dict[str, Tuple[float, float]]


def to_flow_node_specs(nodes_data: List[NodeData], positions: PositionMap) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for node in nodes_data:
        x, y = positions.get(node["id"], (0.0, 0.0))
        specs.append(
            {
                "id": node["id"],
                "pos": (float(x), float(y)),
                "data": {"content": node.get("label", "")},
                "node_type": node.get("type", "default"),
                "source_position": "bottom",
                "target_position": "top",
                "draggable": True,
            }
        )
    return specs


def to_flow_edge_specs(edges_data: List[EdgeData]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for edge in edges_data:
        specs.append(
            {
                "id": edge["id"],
                "source": edge["source"],
                "target": edge["target"],
                "label": edge.get("label", ""),
                "animated": True,
            }
        )
    return specs


def _get_item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def flow_items_to_graph_data(
    flow_nodes: Sequence[Any], flow_edges: Sequence[Any]
) -> Tuple[List[NodeData], List[EdgeData]]:
    nodes: List[NodeData] = []
    for node in flow_nodes:
        raw_data = _get_item_value(node, "data", {})
        label = ""
        if isinstance(raw_data, Mapping):
            label = str(raw_data.get("content", ""))
        elif raw_data is not None:
            label = str(raw_data)

        nodes.append(
            {
                "id": str(_get_item_value(node, "id", "")),
                "label": label,
                "type": str(_get_item_value(node, "node_type", "default")),
            }
        )

    edges: List[EdgeData] = []
    for edge in flow_edges:
        edges.append(
            {
                "id": str(_get_item_value(edge, "id", "")),
                "source": str(_get_item_value(edge, "source", "")),
                "target": str(_get_item_value(edge, "target", "")),
                "label": str(_get_item_value(edge, "label", "") or ""),
            }
        )

    return nodes, edges
