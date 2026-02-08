import pytest

pytest.importorskip("networkx")

from src.mermaid_generator.graph_logic import (
    build_mock_graph,
    calculate_layout_positions,
    export_to_mermaid,
)


def test_build_mock_graph_contains_topic_label():
    graph = build_mock_graph("payment flow")
    labels = [node["label"] for node in graph["nodes"]]
    assert any("payment flow" in label for label in labels)
    assert len(graph["edges"]) == 4


def test_calculate_layout_positions_is_hierarchical_for_dag():
    graph = build_mock_graph("x")
    positions = calculate_layout_positions(graph["nodes"], graph["edges"])
    assert positions["start"][1] < positions["proc1"][1]
    assert positions["proc1"][1] < positions["decide"][1]
    assert positions["end_ok"][1] == positions["end_ng"][1]


def test_calculate_layout_positions_falls_back_on_cycle():
    nodes = [
        {"id": "a", "label": "A", "type": "default"},
        {"id": "b", "label": "B", "type": "default"},
    ]
    edges = [
        {"id": "e1", "source": "a", "target": "b", "label": ""},
        {"id": "e2", "source": "b", "target": "a", "label": ""},
    ]
    positions = calculate_layout_positions(nodes, edges)
    assert set(positions.keys()) == {"a", "b"}
    for coord in positions.values():
        assert isinstance(coord[0], float)
        assert isinstance(coord[1], float)


def test_export_to_mermaid_handles_labeled_and_unlabeled_edges():
    nodes = [
        {"id": "n1", "label": "Start", "type": "input"},
        {"id": "n2", "label": "End", "type": "output"},
    ]
    edges = [
        {"id": "e1", "source": "n1", "target": "n2", "label": "OK"},
        {"id": "e2", "source": "n2", "target": "n1", "label": ""},
    ]
    mermaid = export_to_mermaid(nodes, edges)
    assert 'n1["Start"]' in mermaid
    assert "n1 -->|OK| n2;" in mermaid
    assert "n2 --> n1;" in mermaid
