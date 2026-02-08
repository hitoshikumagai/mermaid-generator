from copy import deepcopy
from typing import Dict, List

GraphData = Dict[str, List[Dict[str, str]]]


def apply_node_properties(
    diagram_type: str, graph_data: GraphData, node_id: str, props: Dict[str, str]
) -> GraphData:
    graph = deepcopy(graph_data)
    for node in graph.get("nodes", []):
        if node.get("id") != node_id:
            continue
        node["label"] = _serialize_node_label(diagram_type, node_id, props)
        break
    return graph


def apply_edge_properties(
    diagram_type: str, graph_data: GraphData, edge_id: str, props: Dict[str, str]
) -> GraphData:
    graph = deepcopy(graph_data)
    for edge in graph.get("edges", []):
        if edge.get("id") != edge_id:
            continue
        edge["label"] = _serialize_edge_label(diagram_type, props)
        break
    return graph


def upsert_class_subclass_relation(graph_data: GraphData, child_id: str, parent_id: str) -> GraphData:
    graph = deepcopy(graph_data)
    target_id = f"extends_{child_id}"
    for edge in graph.get("edges", []):
        if edge.get("id") == target_id or (
            edge.get("source") == child_id and edge.get("label", "").strip() == "extends"
        ) or (
            edge.get("source") == child_id and edge.get("target") == parent_id
        ):
            edge["id"] = target_id
            edge["source"] = child_id
            edge["target"] = parent_id
            edge["label"] = "extends"
            return graph

    graph.setdefault("edges", []).append(
        {"id": target_id, "source": child_id, "target": parent_id, "label": "extends"}
    )
    return graph


def parse_node_properties(diagram_type: str, node_id: str, label: str) -> Dict[str, str]:
    raw = (label or "").strip()
    if diagram_type == "Class":
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        name = lines[0] if lines else node_id
        attributes = ""
        methods = ""
        if len(lines) >= 2:
            attributes = lines[1]
        if len(lines) >= 3:
            methods = lines[2]
        return {"name": name, "attributes": attributes, "methods": methods}

    if diagram_type == "ER":
        # ORDER [PK:id; FK:user_id]
        name = raw
        primary_key = ""
        foreign_key = ""
        if "[" in raw and "]" in raw:
            name = raw.split("[", 1)[0].strip()
            meta = raw.split("[", 1)[1].rsplit("]", 1)[0]
            parts = [part.strip() for part in meta.split(";") if part.strip()]
            for part in parts:
                if part.startswith("PK:"):
                    primary_key = part[len("PK:") :].strip()
                if part.startswith("FK:"):
                    foreign_key = part[len("FK:") :].strip()
        return {"name": name or node_id, "primary_key": primary_key, "foreign_key": foreign_key}

    if diagram_type == "State":
        return {"name": raw or node_id}
    if diagram_type == "Sequence":
        return {"alias": raw or node_id}

    return {"name": raw or node_id}


def parse_edge_properties(diagram_type: str, label: str) -> Dict[str, str]:
    raw = (label or "").strip()

    if diagram_type == "Sequence":
        if ":" in raw:
            message_type, message = raw.split(":", 1)
            return {"message_type": message_type.strip() or "sync", "message": message.strip()}
        return {"message_type": "sync", "message": raw}

    if diagram_type == "State":
        # event [guard]
        if "[" in raw and raw.endswith("]"):
            event, guard = raw.split("[", 1)
            return {"event": event.strip(), "guard": guard[:-1].strip()}
        return {"event": raw, "guard": ""}

    if diagram_type == "ER":
        if " " in raw:
            cardinality, relation = raw.split(" ", 1)
            return {"cardinality": cardinality.strip(), "relation": relation.strip()}
        return {"cardinality": "", "relation": raw}

    if diagram_type == "Class":
        if " " in raw:
            relation_type, relation = raw.split(" ", 1)
            return {"relation_type": relation_type.strip(), "relation": relation.strip()}
        return {"relation_type": raw, "relation": ""}

    return {"label": raw}


def _serialize_node_label(diagram_type: str, node_id: str, props: Dict[str, str]) -> str:
    if diagram_type == "Class":
        name = props.get("name", "").strip() or node_id
        attributes = props.get("attributes", "").strip()
        methods = props.get("methods", "").strip()
        lines = [name]
        if attributes:
            lines.append(attributes)
        if methods:
            lines.append(methods)
        return "\n".join(lines)

    if diagram_type == "ER":
        name = props.get("name", "").strip() or node_id
        pk = props.get("primary_key", "").strip()
        fk = props.get("foreign_key", "").strip()
        meta = []
        if pk:
            meta.append(f"PK:{pk}")
        if fk:
            meta.append(f"FK:{fk}")
        if meta:
            return f"{name} [{' ; '.join(meta).replace(' ; ', '; ')}]"
        return name

    if diagram_type == "State":
        return props.get("name", "").strip() or node_id

    if diagram_type == "Sequence":
        return props.get("alias", "").strip() or node_id

    return props.get("name", "").strip() or node_id


def _serialize_edge_label(diagram_type: str, props: Dict[str, str]) -> str:
    if diagram_type == "Sequence":
        message_type = props.get("message_type", "").strip() or "sync"
        message = props.get("message", "").strip()
        if message:
            return f"{message_type}: {message}"
        return message_type

    if diagram_type == "State":
        event = props.get("event", "").strip()
        guard = props.get("guard", "").strip()
        if guard:
            return f"{event} [{guard}]".strip()
        return event

    if diagram_type == "ER":
        cardinality = props.get("cardinality", "").strip()
        relation = props.get("relation", "").strip()
        if cardinality and relation:
            return f"{cardinality} {relation}"
        return cardinality or relation

    if diagram_type == "Class":
        relation_type = props.get("relation_type", "").strip()
        relation = props.get("relation", "").strip()
        if relation_type and relation:
            return f"{relation_type} {relation}"
        return relation_type or relation

    return props.get("label", "").strip()
