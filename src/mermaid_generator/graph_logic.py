from typing import Dict, List, Tuple

import networkx as nx

NodeData = Dict[str, str]
EdgeData = Dict[str, str]
GraphData = Dict[str, List[Dict[str, str]]]
PositionMap = Dict[str, Tuple[float, float]]


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
    for node in nodes_data:
        graph.add_node(node["id"])
    for edge in edges_data:
        graph.add_edge(edge["source"], edge["target"])

    try:
        levels: Dict[str, int] = {}
        for node_id in nx.topological_sort(graph):
            level = 0
            for pred in graph.predecessors(node_id):
                level = max(level, levels.get(pred, 0) + 1)
            levels[node_id] = level

        positions: PositionMap = {}
        level_counts: Dict[int, int] = {}
        for node in nodes_data:
            node_id = node["id"]
            level = levels.get(node_id, 0)
            count = level_counts.get(level, 0)
            positions[node_id] = (count * 200.0, level * 150.0)
            level_counts[level] = count + 1
        return positions
    except nx.NetworkXUnfeasible:
        # Cyclic graphs use spring layout first, then normalize and separate nodes to avoid overlap.
        raw_positions = nx.spring_layout(graph, k=0.9, iterations=120, seed=42)
        ordered_node_ids = [node["id"] for node in nodes_data]
        return _normalize_and_separate_positions(raw_positions, ordered_node_ids)


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


def _normalize_and_separate_positions(
    raw_positions: Dict[str, Tuple[float, float]], ordered_node_ids: List[str]
) -> PositionMap:
    xs = [float(raw_positions[node_id][0]) for node_id in ordered_node_ids if node_id in raw_positions]
    ys = [float(raw_positions[node_id][1]) for node_id in ordered_node_ids if node_id in raw_positions]
    if not xs or not ys:
        return {node_id: (0.0, 0.0) for node_id in ordered_node_ids}

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    # Scale to pixel-like coordinates first.
    scaled: PositionMap = {}
    for node_id in ordered_node_ids:
        if node_id not in raw_positions:
            scaled[node_id] = (0.0, 0.0)
            continue
        raw_x, raw_y = raw_positions[node_id]
        norm_x = (float(raw_x) - min_x) / span_x
        norm_y = (float(raw_y) - min_y) / span_y
        scaled[node_id] = (norm_x * 800.0, norm_y * 600.0)

    # Then enforce minimum spacing between node boxes.
    min_dx = 180.0
    min_dy = 120.0
    positions: PositionMap = {}
    for node_id in ordered_node_ids:
        base_x, base_y = scaled[node_id]
        x, y = base_x, base_y
        guard = 0
        while any(abs(x - ox) < min_dx and abs(y - oy) < min_dy for ox, oy in positions.values()):
            x += min_dx
            guard += 1
            if guard % 6 == 0:
                x = base_x
                y += min_dy
            if guard > 60:
                break
        positions[node_id] = (float(x), float(y))

    return positions
