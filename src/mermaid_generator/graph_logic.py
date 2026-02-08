import re
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

MERMAID_RESERVED_IDS = {
    "end",
    "subgraph",
    "graph",
    "flowchart",
    "style",
    "linkstyle",
    "class",
    "classdef",
    "click",
}


def build_mock_graph(topic: str) -> GraphData:
    cleaned_topic = _normalize_topic_label(topic)
    return {
        "nodes": [
            {"id": "start", "label": "Start", "type": "input"},
            {"id": "proc1", "label": f"Research: {cleaned_topic}", "type": "default"},
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


def build_structured_flow_graph(
    text: str,
    max_steps: int = 8,
    max_details_per_step: int = 2,
) -> GraphData:
    sections = _extract_flow_sections(
        text=text,
        max_steps=max_steps,
        max_details_per_step=max_details_per_step,
    )
    if len(sections) < 2:
        return build_mock_graph(text)

    nodes: List[Dict[str, str]] = [{"id": "start", "label": "Start", "type": "input"}]
    edges: List[Dict[str, str]] = []
    edge_counter = 1
    previous_main_id = "start"

    for step_index, section in enumerate(sections, start=1):
        main_id = f"step_{step_index}"
        nodes.append({"id": main_id, "label": section["title"], "type": "default"})
        edges.append({"id": f"e{edge_counter}", "source": previous_main_id, "target": main_id, "label": ""})
        edge_counter += 1

        for detail_index, detail in enumerate(section["details"], start=1):
            detail_id = f"{main_id}_detail_{detail_index}"
            nodes.append({"id": detail_id, "label": detail, "type": "default"})
            edges.append({"id": f"e{edge_counter}", "source": main_id, "target": detail_id, "label": "detail"})
            edge_counter += 1

        previous_main_id = main_id

    nodes.append({"id": "end_node", "label": "End", "type": "output"})
    edges.append({"id": f"e{edge_counter}", "source": previous_main_id, "target": "end_node", "label": "done"})
    return {"nodes": nodes, "edges": edges}


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
    id_map: Dict[str, str] = {}
    used_ids = set()

    for index, node in enumerate(nodes_data, start=1):
        raw_id = str(node.get("id", "")).strip() or f"node_{index}"
        if raw_id in id_map:
            continue
        safe_id = _to_mermaid_safe_id(raw_id)
        base_id = safe_id
        suffix = 2
        while safe_id in used_ids:
            safe_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(safe_id)
        id_map[raw_id] = safe_id

    for node in nodes_data:
        raw_id = str(node.get("id", "")).strip()
        safe_id = id_map.get(raw_id, "")
        if not safe_id:
            continue
        label = _sanitize_mermaid_label(node.get("label", ""))
        lines.append(f'    {safe_id}["{label}"];')
    for edge in edges_data:
        source = id_map.get(str(edge.get("source", "")).strip(), "")
        target = id_map.get(str(edge.get("target", "")).strip(), "")
        if not source or not target:
            continue
        label = _sanitize_mermaid_edge_label(edge.get("label", ""))
        if label:
            lines.append(f'    {source} -->|{label}| {target};')
        else:
            lines.append(f'    {source} --> {target};')
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


def _normalize_topic_label(topic: str, max_len: int = 72) -> str:
    raw = (topic or "").strip()
    if not raw:
        return "workflow"

    if "Session Memory:" in raw:
        raw = raw.split("Session Memory:", 1)[0].strip()

    candidates: List[str] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^#+\s*", "", cleaned)
        cleaned = re.sub(r"^\d+[\).]\s*", "", cleaned)
        cleaned = re.sub(r"^[-*]\s*", "", cleaned)
        cleaned = re.sub(r"^>\s*", "", cleaned)
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        if cleaned.startswith("|") and cleaned.endswith("|"):
            continue
        if re.fullmatch(r"[-:| ]+", cleaned):
            continue
        fragments = [frag.strip() for frag in re.split(r"[。.!?]", cleaned) if frag.strip()]
        candidates.extend(fragments or [cleaned])

    chosen = candidates[0] if candidates else "workflow"
    if len(chosen) > max_len:
        chosen = f"{chosen[: max_len - 3].rstrip()}..."
    return chosen


def _extract_flow_sections(
    text: str,
    max_steps: int,
    max_details_per_step: int,
) -> List[Dict[str, List[str]]]:
    raw = (text or "").strip()
    if not raw:
        return []

    if "Session Memory:" in raw:
        raw = raw.split("Session Memory:", 1)[0].strip()

    heading_pattern = re.compile(r"^(?:\d+|[\u2460-\u2473])[\)\.\u3001\uFF0E]?\s*(.+)$")
    hash_heading_pattern = re.compile(r"^#{1,6}\s*(.+)$")
    sections: List[Dict[str, List[str]]] = []
    current: Dict[str, List[str]] = {}

    for raw_line in raw.splitlines():
        line = _clean_section_line(raw_line)
        if not line:
            continue

        title = ""
        heading_match = heading_pattern.match(line)
        hash_heading_match = hash_heading_pattern.match(line)
        if heading_match:
            title = heading_match.group(1).strip()
        elif hash_heading_match:
            title = hash_heading_match.group(1).strip()

        if title:
            if len(sections) >= max_steps:
                continue
            current = {"title": [_truncate_text(title, 56)], "details": []}
            sections.append(current)
            continue

        if not sections:
            continue

        if len(current.get("details", [])) >= max_details_per_step:
            continue

        if _is_section_noise(line):
            continue
        current["details"].append(_truncate_text(line, 68))

    if sections:
        return [{"title": item["title"][0], "details": item["details"]} for item in sections if item["title"]]

    sentences = [s.strip() for s in re.split(r"[。.!?]", raw) if s.strip()]
    normalized = [_truncate_text(_clean_section_line(s), 56) for s in sentences]
    normalized = [item for item in normalized if item]
    return [{"title": item, "details": []} for item in normalized[:max_steps]]


def _clean_section_line(line: str) -> str:
    cleaned = (line or "").strip()
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"^[-*]\s*", "", cleaned)
    cleaned = re.sub(r"^>\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_section_noise(line: str) -> bool:
    if not line:
        return True
    if line.startswith("|") and line.endswith("|"):
        return True
    if re.fullmatch(r"[-:| ]+", line):
        return True
    lowered = line.lower()
    if lowered in {"summary", "background", "requirements", "acceptance criteria"}:
        return True
    return False


def _truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3].rstrip()}..."


def _to_mermaid_safe_id(raw_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw_id or "").strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        normalized = "node"
    if not re.match(r"^[A-Za-z_]", normalized):
        normalized = f"n_{normalized}"
    if normalized.lower() in MERMAID_RESERVED_IDS:
        normalized = f"node_{normalized}"
    return normalized


def _sanitize_mermaid_label(label: str) -> str:
    text = str(label or "")
    text = text.replace("\n", " ")
    text = text.replace('"', '\\"')
    return re.sub(r"\s+", " ", text).strip()


def _sanitize_mermaid_edge_label(label: str) -> str:
    return _sanitize_mermaid_label(label).replace("|", "/")
