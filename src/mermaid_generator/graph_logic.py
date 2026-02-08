from typing import Dict, List, Tuple

import networkx as nx

NodeData = Dict[str, str]
EdgeData = Dict[str, str]
GraphData = Dict[str, List[Dict[str, str]]]
PositionMap = Dict[str, Tuple[float, float]]

LAYER_X_GAP = 240.0
LAYER_Y_GAP = 170.0
PADDING_X = 120.0
PADDING_Y = 80.0


def build_mock_graph(topic: str) -> GraphData:
    return {
        "nodes": [
            {"id": "start", "label": "Start", "type": "input"},
            {"id": "proc1", "label": f"Research: {topic}", "type": "default"},
            {"id": "decide", "label": "Decision", "type": "default"},
            {"id": "end_ok", "label": "Success", "type": "output"},
            {"id": "end_ng", "label": "Fail", "type": "output"},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "proc1", "label": ""},
            {"id": "e2", "source": "proc1", "target": "decide", "label": "Done"},
            {"id": "e3", "source": "decide", "target": "end_ok", "label": "OK"},
            {"id": "e4", "source": "decide", "target": "end_ng", "label": "NG"},
        ],
    }


def calculate_layout_positions(
    nodes_data: List[NodeData], edges_data: List[EdgeData]
) -> PositionMap:
    graph = nx.DiGraph()
    ordered_node_ids = [node["id"] for node in nodes_data]

    for node_id in ordered_node_ids:
        graph.add_node(node_id)
    for edge in edges_data:
        source = edge["source"]
        target = edge["target"]
        if source in graph and target in graph:
            graph.add_edge(source, target)

    if not ordered_node_ids:
        return {}

    return _edge_aware_layered_layout(graph, ordered_node_ids)


def export_to_mermaid(nodes_data: List[NodeData], edges_data: List[EdgeData]) -> str:
    lines = ["graph TD;"]
    for node in nodes_data:
        label = node.get("label", "").replace('"', '\\"')
        lines.append(f'    {node["id"]}["{label}"];')
    for edge in edges_data:
        label = edge.get("label", "").replace('"', '\\"')
        if label:
            lines.append(f'    {edge["source"]} -->|{label}| {edge["target"]};')
        else:
            lines.append(f'    {edge["source"]} --> {edge["target"]};')
    return "\n".join(lines) + "\n"


def count_edge_crossings(edges_data: List[EdgeData], positions: PositionMap) -> int:
    segments = []
    for edge in edges_data:
        source = edge["source"]
        target = edge["target"]
        if source in positions and target in positions:
            segments.append((edge["id"], source, target, positions[source], positions[target]))

    crossings = 0
    for i in range(len(segments)):
        _, s1, t1, p1, p2 = segments[i]
        for j in range(i + 1, len(segments)):
            _, s2, t2, q1, q2 = segments[j]

            # Shared endpoint is not considered a crossing.
            if len({s1, t1, s2, t2}) < 4:
                continue

            if _segments_intersect(p1, p2, q1, q2):
                crossings += 1

    return crossings


def _edge_aware_layered_layout(graph: nx.DiGraph, ordered_node_ids: List[str]) -> PositionMap:
    dag = _make_acyclic(graph, ordered_node_ids)
    levels = _assign_levels(dag, ordered_node_ids)
    layers = _build_layers(levels, ordered_node_ids)

    if not layers:
        return {node_id: (0.0, 0.0) for node_id in ordered_node_ids}

    layers = _reduce_crossings(dag, layers, sweeps=8)

    positions: PositionMap = {}
    for level_idx, layer in enumerate(layers):
        start_x = -((len(layer) - 1) * LAYER_X_GAP) / 2.0
        y = level_idx * LAYER_Y_GAP
        for index, node_id in enumerate(layer):
            positions[node_id] = (start_x + index * LAYER_X_GAP, y)

    positions = _nudge_singleton_layers(dag, layers, positions)
    return _shift_positions_to_positive(positions)


def _make_acyclic(graph: nx.DiGraph, ordered_node_ids: List[str]) -> nx.DiGraph:
    order = {node_id: idx for idx, node_id in enumerate(ordered_node_ids)}
    dag = graph.copy()

    while True:
        try:
            list(nx.topological_sort(dag))
            return dag
        except nx.NetworkXUnfeasible:
            cycle = next(nx.simple_cycles(dag), None)
            if not cycle:
                return dag

            cycle_edges = []
            for idx in range(len(cycle)):
                source = cycle[idx]
                target = cycle[(idx + 1) % len(cycle)]
                if dag.has_edge(source, target):
                    cycle_edges.append((source, target))

            if not cycle_edges:
                return dag

            # Remove the most "backward" edge in original order to keep forward flow.
            edge_to_remove = max(
                cycle_edges,
                key=lambda edge: (
                    order.get(edge[0], 0) - order.get(edge[1], 0),
                    order.get(edge[0], 0),
                    -order.get(edge[1], 0),
                ),
            )
            dag.remove_edge(*edge_to_remove)


def _assign_levels(dag: nx.DiGraph, ordered_node_ids: List[str]) -> Dict[str, int]:
    levels: Dict[str, int] = {node_id: 0 for node_id in ordered_node_ids}

    try:
        topo = list(nx.topological_sort(dag))
    except nx.NetworkXUnfeasible:
        topo = ordered_node_ids

    for node_id in topo:
        preds = list(dag.predecessors(node_id))
        if preds:
            levels[node_id] = max(levels[pred] + 1 for pred in preds)

    return levels


def _build_layers(levels: Dict[str, int], ordered_node_ids: List[str]) -> List[List[str]]:
    layer_map: Dict[int, List[str]] = {}
    for node_id in ordered_node_ids:
        level = levels.get(node_id, 0)
        layer_map.setdefault(level, []).append(node_id)

    return [layer_map[level] for level in sorted(layer_map.keys())]


def _reduce_crossings(dag: nx.DiGraph, layers: List[List[str]], sweeps: int = 6) -> List[List[str]]:
    if len(layers) <= 1:
        return layers

    arranged = [list(layer) for layer in layers]

    for _ in range(sweeps):
        # Downward sweep: order each layer by predecessor barycenter.
        for layer_idx in range(1, len(arranged)):
            arranged[layer_idx] = _order_layer_by_reference(
                arranged[layer_idx],
                arranged[layer_idx - 1],
                lambda node_id: [pred for pred in dag.predecessors(node_id)],
            )

        # Upward sweep: order each layer by successor barycenter.
        for layer_idx in range(len(arranged) - 2, -1, -1):
            arranged[layer_idx] = _order_layer_by_reference(
                arranged[layer_idx],
                arranged[layer_idx + 1],
                lambda node_id: [succ for succ in dag.successors(node_id)],
            )

    return arranged


def _order_layer_by_reference(
    target_layer: List[str],
    reference_layer: List[str],
    neighbor_getter,
) -> List[str]:
    ref_index = {node_id: idx for idx, node_id in enumerate(reference_layer)}
    current_index = {node_id: idx for idx, node_id in enumerate(target_layer)}

    def key(node_id: str) -> Tuple[float, int]:
        neighbors = [n for n in neighbor_getter(node_id) if n in ref_index]
        if neighbors:
            barycenter = sum(ref_index[n] for n in neighbors) / float(len(neighbors))
        else:
            barycenter = float(current_index[node_id])
        return barycenter, current_index[node_id]

    return sorted(target_layer, key=key)


def _shift_positions_to_positive(positions: PositionMap) -> PositionMap:
    if not positions:
        return positions

    min_x = min(pos[0] for pos in positions.values())
    min_y = min(pos[1] for pos in positions.values())

    shift_x = -min_x + PADDING_X if min_x < PADDING_X else 0.0
    shift_y = -min_y + PADDING_Y if min_y < PADDING_Y else 0.0

    return {
        node_id: (float(x + shift_x), float(y + shift_y))
        for node_id, (x, y) in positions.items()
    }


def _nudge_singleton_layers(
    dag: nx.DiGraph, layers: List[List[str]], positions: PositionMap
) -> PositionMap:
    adjusted = dict(positions)

    for level_idx, layer in enumerate(layers):
        if len(layer) != 1:
            continue
        node_id = layer[0]
        preds = [pred for pred in dag.predecessors(node_id) if pred in adjusted]
        succs = [succ for succ in dag.successors(node_id) if succ in adjusted]
        anchors = preds + succs
        if not anchors:
            continue

        # Keep singleton nodes near incident lanes to avoid tiny visual edge crossings.
        anchor_x = sorted(adjusted[anchor][0] for anchor in anchors)
        center_idx = len(anchor_x) // 2
        if len(anchor_x) % 2 == 1:
            target_x = anchor_x[center_idx]
        else:
            target_x = (anchor_x[center_idx - 1] + anchor_x[center_idx]) / 2.0

        _, y = adjusted[node_id]
        adjusted[node_id] = (target_x, y)

    return adjusted


def _segments_intersect(
    p1: Tuple[float, float], p2: Tuple[float, float], q1: Tuple[float, float], q2: Tuple[float, float]
) -> bool:
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if o1 == 0 and _on_segment(p1, q1, p2):
        return True
    if o2 == 0 and _on_segment(p1, q2, p2):
        return True
    if o3 == 0 and _on_segment(q1, p1, q2):
        return True
    if o4 == 0 and _on_segment(q1, p2, q2):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _orientation(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else -1


def _on_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    return (
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
    )
