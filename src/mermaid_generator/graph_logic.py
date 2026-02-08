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
        positions = nx.spring_layout(graph, k=0.6, iterations=60, seed=42)
        return {key: (float(value[0]), float(value[1])) for key, value in positions.items()}


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
