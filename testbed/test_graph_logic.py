import pytest

pytest.importorskip("networkx")

from src.mermaid_generator.graph_logic import (
    build_mock_graph,
    build_structured_flow_graph,
    calculate_layout_positions,
    count_edge_crossings,
    export_to_mermaid,
)


def test_build_mock_graph_contains_topic_label():
    graph = build_mock_graph("payment flow")
    labels = [node["label"] for node in graph["nodes"]]
    assert any("payment flow" in label for label in labels)
    assert len(graph["edges"]) == 4


def test_build_mock_graph_strips_session_memory_and_markdown_noise():
    graph = build_mock_graph(
        "## Email workflow\n"
        "| Action | Rule |\n"
        "| --- | --- |\n"
        "- Process inbox\n\n"
        "Session Memory:\n"
        "- template=EC Purchase Flow (id=ec_purchase, stage=bootstrap)\n"
    )
    proc = next(node for node in graph["nodes"] if node["id"] == "proc1")
    assert "\n" not in proc["label"]
    assert "Session Memory" not in proc["label"]
    assert "template=" not in proc["label"]


def test_build_structured_flow_graph_expands_numbered_sections():
    text = (
        "1. Decide batch schedule\n"
        "Handle email at fixed times.\n"
        "2. Apply two-minute rule\n"
        "Reply immediately if quick.\n"
        "3. Keep folders minimal\n"
        "Use Action and Archive.\n"
        "4. Reuse templates\n"
        "Store common replies.\n"
    )
    graph = build_structured_flow_graph(text)

    labels = [node["label"] for node in graph["nodes"]]
    assert any("Decide batch schedule" in label for label in labels)
    assert any("Apply two-minute rule" in label for label in labels)
    assert any("Keep folders minimal" in label for label in labels)
    assert any("Reuse templates" in label for label in labels)
    assert len(graph["nodes"]) >= 8


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


def test_edge_aware_layout_reduces_crossing_for_bipartite_case():
    nodes = [
        {"id": "a", "label": "A", "type": "input"},
        {"id": "b", "label": "B", "type": "input"},
        {"id": "c", "label": "C", "type": "output"},
        {"id": "d", "label": "D", "type": "output"},
    ]
    edges = [
        {"id": "e1", "source": "a", "target": "d", "label": ""},
        {"id": "e2", "source": "b", "target": "c", "label": ""},
    ]

    positions = calculate_layout_positions(nodes, edges)
    crossings = count_edge_crossings(edges, positions)

    assert crossings == 0


def test_cycle_layout_spreads_positions_without_collapse():
    nodes = [
        {"id": "a", "label": "A", "type": "default"},
        {"id": "b", "label": "B", "type": "default"},
        {"id": "c", "label": "C", "type": "default"},
        {"id": "d", "label": "D", "type": "default"},
    ]
    edges = [
        {"id": "e1", "source": "a", "target": "b", "label": ""},
        {"id": "e2", "source": "b", "target": "c", "label": ""},
        {"id": "e3", "source": "c", "target": "d", "label": ""},
        {"id": "e4", "source": "d", "target": "a", "label": ""},
    ]

    positions = calculate_layout_positions(nodes, edges)
    unique_positions = {(round(x, 3), round(y, 3)) for x, y in positions.values()}

    assert len(unique_positions) == len(nodes)


def test_feedback_loop_layout_aligns_singleton_lanes():
    nodes = [
        {"id": "start", "label": "Submit Request", "type": "input"},
        {"id": "review", "label": "Manager Review", "type": "default"},
        {"id": "approve", "label": "Approved", "type": "output"},
        {"id": "reject", "label": "Rejected", "type": "output"},
        {"id": "rework", "label": "Rework", "type": "default"},
    ]
    edges = [
        {"id": "e1", "source": "start", "target": "review", "label": ""},
        {"id": "e2", "source": "review", "target": "approve", "label": "Approve"},
        {"id": "e3", "source": "review", "target": "reject", "label": "Reject"},
        {"id": "e4", "source": "reject", "target": "rework", "label": "Fix"},
        {"id": "e5", "source": "rework", "target": "review", "label": "Resubmit"},
    ]

    positions = calculate_layout_positions(nodes, edges)

    assert abs(positions["start"][0] - positions["review"][0]) < 1e-6
    assert abs(positions["reject"][0] - positions["rework"][0]) < 1e-6
