import re
from typing import Dict, List, Tuple

from .graph_logic import GraphData, export_to_mermaid

NodeData = Dict[str, str]
EdgeData = Dict[str, str]

START_NODE_ID = "__start__"
END_NODE_ID = "__end__"


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
        match = re.match(r"^([A-Za-z0-9_]+)\s*[-.<>=xo]+>\s*([A-Za-z0-9_]+)$", relation.strip())
        if not match:
            continue
        source = _safe_id(match.group(1))
        target = _safe_id(match.group(2))
        for node_id in (source, target):
            if node_id and node_id not in node_ids:
                node_ids.add(node_id)
                nodes.append({"id": node_id, "label": node_id, "type": "default"})
        edges.append(
            {
                "id": f"e{len(edges) + 1}",
                "source": source,
                "target": target,
                "label": label.strip(),
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
            r"^([A-Za-z][A-Za-z0-9_]*)\s+\S+\s+([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*)$",
            line,
        )
        if not relation:
            continue
        source = _safe_id(relation.group(1))
        target = _safe_id(relation.group(2))
        label = relation.group(3).strip()
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

        relation = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*)\s+[-.<|*o]+>\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*:\s*(.*))?$",
            line,
        )
        if not relation:
            relation = re.match(
                r"^([A-Za-z_][A-Za-z0-9_]*)\s+<[-.<|*o]+\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*:\s*(.*))?$",
                line,
            )
            if relation:
                source = _safe_id(relation.group(2))
                target = _safe_id(relation.group(1))
                label = (relation.group(3) or "").strip()
            else:
                continue
        else:
            source = _safe_id(relation.group(1))
            target = _safe_id(relation.group(2))
            label = (relation.group(3) or "").strip()

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
        node_id, dependency = _parse_gantt_spec(spec, label)
        if node_id not in node_ids:
            node_ids.add(node_id)
            nodes.append({"id": node_id, "label": label or node_id, "type": "default"})
        if dependency:
            if dependency not in node_ids:
                node_ids.add(dependency)
                nodes.append({"id": dependency, "label": dependency, "type": "default"})
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
        label = edge.get("label", "").strip() or "message"
        lines.append(f"    {edge['source']}->>{edge['target']}: {label}")
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
        label = edge.get("label", "").strip() or "relates_to"
        lines.append(f"    {edge['source']} ||--o{{ {edge['target']} : {label}")
    if not edges and len(nodes) >= 2:
        lines.append(f"    {nodes[0]['id']} ||--o{{ {nodes[1]['id']} : relates_to")
    return "\n".join(lines) + "\n"


def _class_to_mermaid(nodes: List[NodeData], edges: List[EdgeData]) -> str:
    lines = ["classDiagram"]
    for node in nodes:
        lines.append(f"    class {node['id']}")
    for edge in edges:
        label = edge.get("label", "").strip()
        if label:
            lines.append(f"    {edge['source']} --> {edge['target']} : {label}")
        else:
            lines.append(f"    {edge['source']} --> {edge['target']}")
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
        node_id = _safe_id(node["id"])
        label = node.get("label", node_id) or node_id
        dependency = dependency_by_target.get(node_id, "")
        if dependency:
            lines.append(f"    {label} :{node_id}, after {dependency}, 3d")
        elif not has_start_task:
            lines.append(f"    {label} :{node_id}, 2026-02-10, 3d")
            has_start_task = True
        else:
            previous = _safe_id(nodes[index - 1]["id"])
            lines.append(f"    {label} :{node_id}, after {previous}, 3d")
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


def _parse_gantt_spec(spec: str, label: str) -> Tuple[str, str]:
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    node_id = ""
    dependency = ""

    if tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tokens[0]):
        node_id = tokens[0]
    if not node_id:
        node_id = _safe_id(label) or f"task_{abs(hash(spec)) % 10000}"

    dep_match = re.search(r"\bafter\s+([A-Za-z_][A-Za-z0-9_]*)", spec)
    if dep_match:
        dependency = dep_match.group(1)

    return node_id, dependency
