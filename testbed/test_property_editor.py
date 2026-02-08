from src.mermaid_generator.property_editor import (
    apply_edge_properties,
    apply_node_properties,
    parse_edge_properties,
    parse_node_properties,
    upsert_class_subclass_relation,
)


def _base_graph():
    return {
        "nodes": [
            {"id": "A", "label": "A", "type": "default"},
            {"id": "B", "label": "B", "type": "default"},
        ],
        "edges": [{"id": "e1", "source": "A", "target": "B", "label": ""}],
    }


def test_class_subclass_relation_can_be_upserted_without_manual_mermaid():
    graph = _base_graph()
    updated = upsert_class_subclass_relation(graph, child_id="A", parent_id="B")

    extends_edges = [e for e in updated["edges"] if e["source"] == "A" and e["target"] == "B"]
    assert extends_edges
    assert extends_edges[0]["label"] == "extends"


def test_sequence_edge_properties_are_serialized_in_label():
    graph = _base_graph()
    updated = apply_edge_properties(
        "Sequence",
        graph,
        edge_id="e1",
        props={"message_type": "async", "message": "notify"},
    )
    edge = updated["edges"][0]
    assert edge["label"] == "async: notify"
    parsed = parse_edge_properties("Sequence", edge["label"])
    assert parsed["message_type"] == "async"
    assert parsed["message"] == "notify"


def test_state_edge_guard_is_preserved():
    graph = _base_graph()
    updated = apply_edge_properties(
        "State",
        graph,
        edge_id="e1",
        props={"event": "approve", "guard": "score > 80"},
    )
    edge = updated["edges"][0]
    assert edge["label"] == "approve [score > 80]"
    parsed = parse_edge_properties("State", edge["label"])
    assert parsed["event"] == "approve"
    assert parsed["guard"] == "score > 80"


def test_class_node_properties_are_serialized():
    graph = _base_graph()
    updated = apply_node_properties(
        "Class",
        graph,
        node_id="A",
        props={"name": "OrderService", "attributes": "id:int", "methods": "createOrder()"},
    )
    node = next(n for n in updated["nodes"] if n["id"] == "A")
    assert "OrderService" in node["label"]
    assert "id:int" in node["label"]
    assert "createOrder()" in node["label"]
    parsed = parse_node_properties("Class", node["id"], node["label"])
    assert parsed["name"] == "OrderService"


def test_er_node_properties_include_pk_fk():
    graph = _base_graph()
    updated = apply_node_properties(
        "ER",
        graph,
        node_id="A",
        props={"name": "ORDER", "primary_key": "id", "foreign_key": "user_id"},
    )
    node = next(n for n in updated["nodes"] if n["id"] == "A")
    assert "ORDER" in node["label"]
    assert "PK:id" in node["label"]
    assert "FK:user_id" in node["label"]
    parsed = parse_node_properties("ER", node["id"], node["label"])
    assert parsed["primary_key"] == "id"


def test_gantt_node_properties_update_metadata_and_dependency_edge():
    graph = {
        "nodes": [
            {"id": "a1", "label": "Scope Freeze", "type": "default"},
            {"id": "b1", "label": "Implementation", "type": "default"},
        ],
        "edges": [],
    }
    updated = apply_node_properties(
        "Gantt",
        graph,
        node_id="a1",
        props={
            "name": "Planning",
            "task_id": "plan_1",
            "dependency": "b1",
            "start": "2026-02-10",
            "duration": "3d",
            "flags": "crit, done",
        },
    )

    node = next(n for n in updated["nodes"] if n["id"] == "plan_1")
    assert node["label"] == "Planning"
    assert node["metadata"]["task_id"] == "plan_1"
    assert node["metadata"]["dependency"] == "b1"
    assert node["metadata"]["start"] == "2026-02-10"
    assert node["metadata"]["duration"] == "3d"
    assert node["metadata"]["flags"] == "crit, done"

    edge = next(e for e in updated["edges"] if e["target"] == "plan_1")
    assert edge["source"] == "b1"
    assert edge["label"] == "after"

    parsed = parse_node_properties("Gantt", node["id"], node["label"], node.get("metadata"))
    assert parsed["task_id"] == "plan_1"
    assert parsed["dependency"] == "b1"
