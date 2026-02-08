from typing import Any, Dict, List, Mapping, Sequence, Tuple

NodeData = Dict[str, str]
EdgeData = Dict[str, str]
PositionMap = Dict[str, Tuple[float, float]]


def to_flow_node_specs(
    nodes_data: List[NodeData], positions: PositionMap, edges_data: List[EdgeData] = None
) -> List[Dict[str, Any]]:
    edges_data = edges_data or []
    source_positions, target_positions = _resolve_node_handle_positions(
        nodes_data, positions, edges_data
    )

    specs: List[Dict[str, Any]] = []
    for node in nodes_data:
        node_id = node["id"]
        x, y = positions.get(node["id"], (0.0, 0.0))
        specs.append(
            {
                "id": node_id,
                "pos": (float(x), float(y)),
                "data": {"content": node.get("label", "")},
                "node_type": node.get("type", "default"),
                "source_position": source_positions.get(node_id, "bottom"),
                "target_position": target_positions.get(node_id, "top"),
                "draggable": True,
            }
        )
    return specs


def to_flow_edge_specs(edges_data: List[EdgeData], positions: PositionMap = None) -> List[Dict[str, Any]]:
    positions = positions or {}
    specs: List[Dict[str, Any]] = []
    for edge in edges_data:
        source = edge["source"]
        target = edge["target"]
        edge_type = "smoothstep"
        if source in positions and target in positions:
            sy = positions[source][1]
            ty = positions[target][1]
            if ty < sy:
                edge_type = "step"
            elif abs(ty - sy) < 1e-6:
                edge_type = "straight"

        specs.append(
            {
                "id": edge["id"],
                "source": source,
                "target": target,
                "label": edge.get("label", ""),
                "animated": True,
                "edge_type": edge_type,
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


def _resolve_node_handle_positions(
    nodes_data: List[NodeData], positions: PositionMap, edges_data: List[EdgeData]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    node_ids = {node["id"] for node in nodes_data}
    outgoing: Dict[str, List[Tuple[float, float]]] = {node_id: [] for node_id in node_ids}
    incoming: Dict[str, List[Tuple[float, float]]] = {node_id: [] for node_id in node_ids}

    for edge in edges_data:
        source = edge["source"]
        target = edge["target"]
        if source not in positions or target not in positions:
            continue
        if source not in node_ids or target not in node_ids:
            continue
        sx, sy = positions[source]
        tx, ty = positions[target]
        dx = tx - sx
        dy = ty - sy
        outgoing[source].append((dx, dy))
        incoming[target].append((dx, dy))

    source_positions = {
        node_id: _choose_side_from_outgoing(outgoing[node_id]) for node_id in node_ids
    }
    target_positions = {
        node_id: _choose_side_from_incoming(incoming[node_id]) for node_id in node_ids
    }
    return source_positions, target_positions


def _choose_side_from_outgoing(vectors: List[Tuple[float, float]]) -> str:
    if not vectors:
        return "bottom"

    scores = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
    for dx, dy in vectors:
        if abs(dx) > abs(dy):
            side = "right" if dx >= 0 else "left"
            scores[side] += abs(dx)
        else:
            side = "bottom" if dy >= 0 else "top"
            scores[side] += abs(dy)

    return max(("bottom", "right", "left", "top"), key=lambda side: scores[side])


def _choose_side_from_incoming(vectors: List[Tuple[float, float]]) -> str:
    if not vectors:
        return "top"

    from_above = any(dy > 0 and abs(dy) >= abs(dx) for dx, dy in vectors)
    from_below = any(dy < 0 and abs(dy) >= abs(dx) for dx, dy in vectors)
    if from_above and from_below:
        # Keep primary top-down readability when feedback edges exist.
        return "top"

    scores = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
    for dx, dy in vectors:
        if abs(dx) > abs(dy):
            side = "left" if dx >= 0 else "right"
            scores[side] += abs(dx)
        else:
            side = "top" if dy >= 0 else "bottom"
            scores[side] += abs(dy)

    return max(("top", "left", "right", "bottom"), key=lambda side: scores[side])
