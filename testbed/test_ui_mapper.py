from types import SimpleNamespace

from src.mermaid_generator.ui_mapper import (
    flow_items_to_graph_data,
    to_flow_edge_specs,
    to_flow_node_specs,
)


def test_to_flow_node_specs_maps_position_and_defaults():
    nodes = [
        {"id": "n1", "label": "Start", "type": "input"},
        {"id": "n2", "label": "Next", "type": "default"},
    ]
    positions = {"n1": (10.5, 20.0)}
    specs = to_flow_node_specs(nodes, positions)

    assert specs[0]["id"] == "n1"
    assert specs[0]["pos"] == (10.5, 20.0)
    assert specs[0]["data"]["content"] == "Start"
    assert specs[0]["node_type"] == "input"
    assert specs[1]["pos"] == (0.0, 0.0)
    assert specs[1]["node_type"] == "default"


def test_to_flow_edge_specs_preserves_labels_and_defaults():
    edges = [
        {"id": "e1", "source": "n1", "target": "n2", "label": "OK"},
        {"id": "e2", "source": "n2", "target": "n3"},
    ]
    specs = to_flow_edge_specs(edges)

    assert specs[0]["label"] == "OK"
    assert specs[1]["label"] == ""
    assert specs[0]["animated"] is True
    assert specs[0]["edge_type"] == "smoothstep"


def test_flow_items_to_graph_data_accepts_objects_and_dicts():
    flow_nodes = [
        SimpleNamespace(id="n1", data={"content": "Start"}, node_type="input"),
        {"id": "n2", "data": {"content": "End"}, "node_type": "output"},
    ]
    flow_edges = [
        SimpleNamespace(id="e1", source="n1", target="n2", label="OK"),
        {"id": "e2", "source": "n2", "target": "n1", "label": None},
    ]

    nodes, edges = flow_items_to_graph_data(flow_nodes, flow_edges)

    assert nodes == [
        {"id": "n1", "label": "Start", "type": "input"},
        {"id": "n2", "label": "End", "type": "output"},
    ]
    assert edges == [
        {"id": "e1", "source": "n1", "target": "n2", "label": "OK"},
        {"id": "e2", "source": "n2", "target": "n1", "label": ""},
    ]


def test_to_flow_node_specs_uses_directional_handles_for_feedback_loop():
    nodes = [
        {"id": "start", "label": "Start", "type": "input"},
        {"id": "review", "label": "Review", "type": "default"},
        {"id": "reject", "label": "Reject", "type": "output"},
        {"id": "rework", "label": "Rework", "type": "default"},
    ]
    positions = {
        "start": (240.0, 80.0),
        "review": (240.0, 250.0),
        "reject": (360.0, 420.0),
        "rework": (360.0, 590.0),
    }
    edges = [
        {"id": "e0", "source": "start", "target": "review", "label": ""},
        {"id": "e1", "source": "review", "target": "reject", "label": "Reject"},
        {"id": "e2", "source": "reject", "target": "rework", "label": "Fix"},
        {"id": "e3", "source": "rework", "target": "review", "label": "Resubmit"},
    ]

    specs = {spec["id"]: spec for spec in to_flow_node_specs(nodes, positions, edges)}

    assert specs["review"]["source_position"] == "bottom"
    assert specs["review"]["target_position"] == "top"
    assert specs["rework"]["source_position"] == "top"
    assert specs["rework"]["target_position"] == "top"


def test_to_flow_edge_specs_marks_back_edges_as_step():
    edges = [
        {"id": "e1", "source": "a", "target": "b", "label": ""},
        {"id": "e2", "source": "b", "target": "c", "label": ""},
    ]
    positions = {
        "a": (100.0, 100.0),
        "b": (200.0, 200.0),
        "c": (150.0, 120.0),
    }

    specs = to_flow_edge_specs(edges, positions)
    specs_by_id = {spec["id"]: spec for spec in specs}

    assert specs_by_id["e1"]["edge_type"] == "smoothstep"
    assert specs_by_id["e2"]["edge_type"] == "step"
