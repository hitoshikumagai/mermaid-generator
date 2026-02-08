import re
from typing import Dict, List, Tuple

from .graph_logic import GraphData, export_to_mermaid

NodeData = Dict[str, str]
EdgeData = Dict[str, str]

START_NODE_ID = "__start__"
END_NODE_ID = "__end__"

SEQUENCE_ARROW_TO_TYPE = {
    "->>": "sync",
    "-->>": "return",
    "-)": "async",
    "--)": "async",
}

CLASS_ARROW_RULES = [
    ("<|--", "extends", True),
    ("--|>", "extends", False),
    ("<|..", "implements", True),
    ("..|>", "implements", False),
    ("<--", "association", True),
    ("-->", "association", False),
    ("<..", "dependency", True),
    ("..>", "dependency", False),
    ("*--", "composition", False),
    ("o--", "aggregation", False),
]

CLASS_RELATION_TO_ARROW = {
    "extends": "--|>",
    "implements": "..|>",
    "composition": "*--",
    "aggregation": "o--",
    "dependency": "..>",
    "association": "-->",
}


def parse_mermaid_to_graph(diagram_type: str, mermaid_code: str) -> GraphData:
    code = (mermaid_code or "").strip()
    if diagram_type == "Flowchart":
        # Flowchart parsing is not required for #8; keep a blank-safe fallback.
        return _default_graph("Flowchart")
    if diagram_type == "Sequence":
        return _parse_sequence(code)
    if diagram_type == "State":
        return _parse_state(code)
    if diagram_type == "ER":
        return _parse_er(code)
    if diagram_type == "Class":
        return _parse_class(code)
    if diagram_type == "Gantt":
        return _parse_gantt(code)
    return _default_graph(diagram_type)


def graph_to_mermaid(diagram_type: str, graph_data: GraphData) -> str:
    nodes = list(graph_data.get("nodes", []))
    edges = list(graph_data.get("edges", []))

    if diagram_type == "Flowchart":
        return export_to_mermaid(nodes, edges)
    if diagram_type == "Sequence":
        return _sequence_to_mermaid(nodes, edges)
    if diagram_type == "State":
        return _state_to_mermaid(nodes, edges)
    if diagram_type == "ER":
        return _er_to_mermaid(nodes, edges)
    if diagram_type == "Class":
        return _class_to_mermaid(nodes, edges)
    if diagram_type == "Gantt":
        return _gantt_to_mermaid(nodes, edges)

    return export_to_mermaid(nodes, edges)


def _parse_sequence(code: str) -> GraphData:
    nodes: List[NodeData] = []
    edges: List[EdgeData] = []
    node_ids = set()

    for raw in code.splitlines():
        line = raw.strip()
        if not line or line == "sequenceDiagram":
            continue
        if line.startswith("participant "):
            body = line[len("participant ") :].strip()
            if " as " in body:
                node_id, label = body.split(" as ", 1)
            else:
                node_id, label = body, body
            node_id = _safe_id(node_id)
            if node_id and node_id not in node_ids:
                node_ids.add(node_id)
                nodes.append({"id": node_id, "label": label.strip() or node_id, "type": "default"})
            continue

        if ":" not in line:
            continue
        relation, label = line.split(":", 1)
        match = re.match(
            r"^([A-Za-z0-9_]+)\s*([-.<>=xo]+\>|[-.<>xo]+\))\s*([A-Za-z0-9_]+)$",
            relation.strip(),
        )
        if not match:
            continue
        source = _safe_id(match.group(1))
        arrow = match.group(2).strip()
        target = _safe_id(match.group(3))
        for node_id in (source, target):
            if node_id and node_id not in node_ids:
                node_ids.add(node_id)
                nodes.append({"id": node_id, "label": node_id, "type": "default"})
        label_text = label.strip()
        message_type = SEQUENCE_ARROW_TO_TYPE.get(arrow, "sync")
        if message_type == "sync":
            edge_label = label_text
        else:
            edge_label = f"{message_type}: {label_text}".strip(": ")
        edges.append(
            {
                "id": f"e{len(edges) + 1}",
                "source": source,
                "target": target,
                "label": edge_label,
            }
        )

    if not nodes:
        return _default_graph("Sequence")
    return {"nodes": nodes, "edges": edges}


def _parse_state(code: str) -> GraphData:
    nodes: List[NodeData] = []
    edges: List[EdgeData] = []
    node_ids = set()

    for raw in code.splitlines():
        line = raw.strip()
        if not line or line.startswith("stateDiagram"):
            continue
        match = re.match(
            r"^(\[\*\]|[A-Za-z0-9_]+)\s*-->\s*(\[\*\]|[A-Za-z0-9_]+)(?:\s*:\s*(.*))?$",
            line,
        )
        if not match:
            continue
        source = _state_node_id(match.group(1), is_source=True)
        target = _state_node_id(match.group(2), is_source=False)
        label = (match.group(3) or "").strip()
        for node_id in (source, target):
            if node_id not in node_ids:
                node_ids.add(node_id)
                nodes.append({"id": node_id, "label": _state_node_label(node_id), "type": "default"})
        edges.append({"id": f"e{len(edges) + 1}", "source": source, "target": target, "label": label})

    if not nodes:
        return _default_graph("State")
    return {"nodes": nodes, "edges": edges}


def _parse_er(code: str) -> GraphData:
    nodes: List[NodeData] = []
    edges: List[EdgeData] = []
    node_ids = set()

    for raw in code.splitlines():
        line = raw.strip()
        if not line or line == "erDiagram":
            continue

        entity_block = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s*\{$", line)
        if entity_block:
            node_id = _safe_id(entity_block.group(1))
            if node_id not in node_ids:
                node_ids.add(node_id)
                nodes.append({"id": node_id, "label": node_id, "type": "default"})
            continue

        relation = re.match(
            r"^([A-Za-z][A-Za-z0-9_]*)\s+([|o}{]{1,2}--[|o}{]{1,2})\s+([A-Za-z][A-Za-z0-9_]*)(?:\s*:\s*(.*))?$",
            line,
        )
        if not relation:
            continue
        source = _safe_id(relation.group(1))
        cardinality = relation.group(2).strip()
        target = _safe_id(relation.group(3))
        relation_label = (relation.group(4) or "").strip()
        label = f"{cardinality} {relation_label}".strip()
        for node_id in (source, target):
            if node_id not in node_ids:
                node_ids.add(node_id)
                nodes.append({"id": node_id, "label": node_id, "type": "default"})
        edges.append({"id": f"e{len(edges) + 1}", "source": source, "target": target, "label": label})

    if not nodes:
        return _default_graph("ER")
    return {"nodes": nodes, "edges": edges}


def _parse_class(code: str) -> GraphData:
    nodes: List[NodeData] = []
    edges: List[EdgeData] = []
    node_ids = set()

    for raw in code.splitlines():
        line = raw.strip()
        if not line or line == "classDiagram":
            continue
        class_decl = re.match(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if class_decl:
            node_id = _safe_id(class_decl.group(1))
            if node_id not in node_ids:
                node_ids.add(node_id)
                nodes.append({"id": node_id, "label": node_id, "type": "default"})
            continue

        parsed = _parse_class_relation_line(line)
        if not parsed:
            continue
        source, target, relation_type, relation_text = parsed
        label = relation_type if not relation_text else f"{relation_type} {relation_text}"

        for node_id in (source, target):
            if node_id not in node_ids:
                node_ids.add(node_id)
                nodes.append({"id": node_id, "label": node_id, "type": "default"})
        edges.append({"id": f"e{len(edges) + 1}", "source": source, "target": target, "label": label})

    if not nodes:
        return _default_graph("Class")
    return {"nodes": nodes, "edges": edges}


def _parse_gantt(code: str) -> GraphData:
    nodes: List[NodeData] = []
    edges: List[EdgeData] = []
    node_ids = set()

    for raw in code.splitlines():
        line = raw.strip()
        if (
            not line
            or line == "gantt"
            or line.startswith("title ")
            or line.startswith("dateFormat ")
            or line.startswith("section ")
        ):
            continue
        if ":" not in line:
            continue
        label, spec = [part.strip() for part in line.split(":", 1)]
        node_id, metadata = _parse_gantt_spec(spec, label)
        dependency = metadata.get("dependency", "")
        if node_id not in node_ids:
            node_ids.add(node_id)
            node = {"id": node_id, "label": label or node_id, "type": "default"}
            if metadata:
                node["metadata"] = metadata
            nodes.append(node)
        if dependency:
            if dependency not in node_ids:
                node_ids.add(dependency)
                nodes.append(
                    {
                        "id": dependency,
                        "label": dependency,
                        "type": "default",
                        "metadata": {"task_id": dependency},
                    }
                )
            edges.append(
                {
                    "id": f"e{len(edges) + 1}",
                    "source": dependency,
                    "target": node_id,
                    "label": "after",
                }
            )

    if not nodes:
        return _default_graph("Gantt")
    return {"nodes": nodes, "edges": edges}


def _sequence_to_mermaid(nodes: List[NodeData], edges: List[EdgeData]) -> str:
    lines = ["sequenceDiagram"]
    for node in nodes:
        node_id = node["id"]
        label = node.get("label", node_id)
        if label != node_id:
            lines.append(f"    participant {node_id} as {label}")
        else:
            lines.append(f"    participant {node_id}")
    for edge in edges:
        message_type, message = _split_sequence_edge_label(edge.get("label", ""))
        arrow = _sequence_type_to_arrow(message_type)
        lines.append(f"    {edge['source']}{arrow}{edge['target']}: {message or 'message'}")
    return "\n".join(lines) + "\n"


def _state_to_mermaid(nodes: List[NodeData], edges: List[EdgeData]) -> str:
    lines = ["stateDiagram-v2"]
    for edge in edges:
        source = _state_symbol(edge["source"])
        target = _state_symbol(edge["target"])
        label = edge.get("label", "").strip()
        if label:
            lines.append(f"    {source} --> {target} : {label}")
        else:
            lines.append(f"    {source} --> {target}")
    if len(edges) == 0 and nodes:
        lines.append(f"    [*] --> {_state_symbol(nodes[0]['id'])}")
    return "\n".join(lines) + "\n"


def _er_to_mermaid(nodes: List[NodeData], edges: List[EdgeData]) -> str:
    lines = ["erDiagram"]
    for edge in edges:
        cardinality, relation = _split_er_edge_label(edge.get("label", ""))
        lines.append(f"    {edge['source']} {cardinality} {edge['target']} : {relation}")
    if not edges and len(nodes) >= 2:
        lines.append(f"    {nodes[0]['id']} ||--o{{ {nodes[1]['id']} : relates_to")
    return "\n".join(lines) + "\n"


def _class_to_mermaid(nodes: List[NodeData], edges: List[EdgeData]) -> str:
    lines = ["classDiagram"]
    for node in nodes:
        lines.append(f"    class {node['id']}")
    for edge in edges:
        relation_type, relation_text = _split_class_edge_label(edge.get("label", ""))
        arrow = CLASS_RELATION_TO_ARROW.get(relation_type, "-->")
        if relation_text:
            lines.append(f"    {edge['source']} {arrow} {edge['target']} : {relation_text}")
        else:
            lines.append(f"    {edge['source']} {arrow} {edge['target']}")
    return "\n".join(lines) + "\n"


def _gantt_to_mermaid(nodes: List[NodeData], edges: List[EdgeData]) -> str:
    lines = [
        "gantt",
        "    title Canvas Plan",
        "    dateFormat  YYYY-MM-DD",
        "    section Main",
    ]
    dependency_by_target: Dict[str, str] = {}
    for edge in edges:
        dependency_by_target.setdefault(edge["target"], edge["source"])

    has_start_task = False
    for index, node in enumerate(nodes):
        metadata = node.get("metadata")
        meta = metadata if isinstance(metadata, dict) else {}

        node_id = _safe_id(str(meta.get("task_id", "") or node["id"]))
        label = node.get("label", node_id) or node_id

        dependency = _safe_id(str(meta.get("dependency", "") or dependency_by_target.get(node_id, "")))
        start = str(meta.get("start", "") or "").strip()
        end = str(meta.get("end", "") or "").strip()
        duration = str(meta.get("duration", "") or "").strip()
        raw_flags = str(meta.get("flags", "") or "")
        flags = [part.strip().lower() for part in raw_flags.split(",") if part.strip()]

        tokens = []
        for flag in flags:
            if flag in {"active", "done", "crit", "milestone"}:
                tokens.append(flag)
        tokens.append(node_id)
        if dependency:
            tokens.append(f"after {dependency}")

        if start and end:
            tokens.extend([start, end])
        elif start:
            tokens.extend([start, duration or "3d"])
        elif end and duration:
            tokens.extend([end, duration])
        elif duration:
            tokens.append(duration)
        elif dependency:
            tokens.append("3d")
        elif not has_start_task:
            tokens.extend(["2026-02-10", "3d"])
            has_start_task = True
        else:
            previous = _safe_id(str(nodes[index - 1].get("id", "")))
            tokens.extend([f"after {previous}", "3d"])

        lines.append(f"    {label} :{', '.join(tokens)}")
    return "\n".join(lines) + "\n"


def _default_graph(diagram_type: str) -> GraphData:
    label = f"{diagram_type} Diagram"
    return {
        "nodes": [
            {"id": "n1", "label": label, "type": "default"},
            {"id": "n2", "label": "Step 2", "type": "default"},
        ],
        "edges": [{"id": "e1", "source": "n1", "target": "n2", "label": ""}],
    }


def _safe_id(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", raw):
        return raw
    raw = re.sub(r"[^A-Za-z0-9_]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    if not raw:
        raw = "node"
    if not re.match(r"^[A-Za-z_]", raw):
        raw = "n_" + raw
    return raw


def _state_node_id(symbol: str, is_source: bool) -> str:
    if symbol == "[*]":
        return START_NODE_ID if is_source else END_NODE_ID
    return _safe_id(symbol)


def _state_node_label(node_id: str) -> str:
    if node_id == START_NODE_ID:
        return "Start"
    if node_id == END_NODE_ID:
        return "End"
    return node_id


def _state_symbol(node_id: str) -> str:
    if node_id == START_NODE_ID or node_id == END_NODE_ID:
        return "[*]"
    return node_id


def _parse_gantt_spec(spec: str, label: str) -> Tuple[str, Dict[str, str]]:
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    node_id = ""
    dependency = ""
    start = ""
    end = ""
    duration = ""
    flags: List[str] = []
    known_flags = {"active", "done", "crit", "milestone"}

    for token in tokens:
        lowered = token.lower()
        if lowered in known_flags:
            if lowered not in flags:
                flags.append(lowered)
            continue
        if lowered.startswith("after "):
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token) and not node_id:
            node_id = token
            continue
        if re.match(r"^\d{4}-\d{2}-\d{2}$", token):
            if not start:
                start = token
            elif not end:
                end = token
            continue
        if re.match(r"^\d+[smhdwMy]$", token):
            duration = token
            continue

    if not node_id:
        node_id = _safe_id(label) or f"task_{abs(hash(spec)) % 10000}"

    dep_match = re.search(r"\bafter\s+([A-Za-z_][A-Za-z0-9_]*)", spec)
    if dep_match:
        dependency = dep_match.group(1)

    metadata: Dict[str, str] = {"task_id": node_id}
    if dependency:
        metadata["dependency"] = dependency
    if start:
        metadata["start"] = start
    if end:
        metadata["end"] = end
    if duration:
        metadata["duration"] = duration
    if flags:
        metadata["flags"] = ", ".join(flags)

    return node_id, metadata


def _split_sequence_edge_label(label: str) -> Tuple[str, str]:
    raw = (label or "").strip()
    if ":" in raw:
        maybe_type, message = raw.split(":", 1)
        message_type = maybe_type.strip().lower()
        if message_type in {"sync", "async", "return"}:
            return message_type, message.strip()
    return "sync", raw


def _sequence_type_to_arrow(message_type: str) -> str:
    normalized = (message_type or "sync").strip().lower()
    if normalized == "return":
        return "-->>"
    if normalized == "async":
        return "-)"
    return "->>"


def _split_er_edge_label(label: str) -> Tuple[str, str]:
    raw = (label or "").strip()
    if raw:
        parts = raw.split(" ", 1)
        if len(parts) == 2 and "--" in parts[0]:
            return parts[0], parts[1].strip() or "relates_to"
    return "||--o{", raw or "relates_to"


def _parse_class_relation_line(line: str):
    relation_text = ""
    relation_part = line
    if ":" in line:
        relation_part, relation_text = line.split(":", 1)
        relation_text = relation_text.strip()
    relation_part = relation_part.strip()

    for arrow, relation_type, reverse in CLASS_ARROW_RULES:
        token = f" {arrow} "
        if token not in relation_part:
            continue
        left, right = relation_part.split(token, 1)
        left_id = _safe_id(left.strip())
        right_id = _safe_id(right.strip())
        if not left_id or not right_id:
            return None
        if reverse:
            source = right_id
            target = left_id
        else:
            source = left_id
            target = right_id
        return source, target, relation_type, relation_text
    return None


def _split_class_edge_label(label: str) -> Tuple[str, str]:
    raw = (label or "").strip()
    if not raw:
        return "association", ""
    if " " in raw:
        relation_type, relation_text = raw.split(" ", 1)
        relation_type = relation_type.strip().lower()
        if relation_type in CLASS_RELATION_TO_ARROW:
            return relation_type, relation_text.strip()
    relation_type = raw.lower()
    if relation_type in CLASS_RELATION_TO_ARROW:
        return relation_type, ""
    return "association", raw
