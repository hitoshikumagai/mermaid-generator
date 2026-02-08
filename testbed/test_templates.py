import pytest

pytest.importorskip("networkx")

from src.mermaid_generator.templates import (
    DIAGRAM_TYPES,
    get_flowchart_template,
    get_mermaid_template,
    list_flowchart_templates,
    list_mermaid_templates,
)
from src.mermaid_generator.graph_logic import calculate_layout_positions


def test_flowchart_templates_include_default_ec_purchase():
    templates = list_flowchart_templates()
    ids = [template["id"] for template in templates]
    assert "ec_purchase" in ids


def test_get_flowchart_template_returns_graph_structure():
    template = get_flowchart_template("approval")
    assert template["name"]
    assert template["graph"]["nodes"]
    assert template["graph"]["edges"]


def test_mermaid_templates_exist_for_common_types():
    assert "Flowchart" in DIAGRAM_TYPES
    assert "Sequence" in DIAGRAM_TYPES
    assert list_mermaid_templates("Sequence")


def test_get_mermaid_template_returns_code():
    templates = list_mermaid_templates("Sequence")
    template_id = templates[0]["id"]
    code = get_mermaid_template("Sequence", template_id)
    assert "sequenceDiagram" in code


def test_flowchart_templates_layout_has_no_overlapping_positions():
    for template in list_flowchart_templates():
        graph = get_flowchart_template(template["id"])["graph"]
        positions = calculate_layout_positions(graph["nodes"], graph["edges"])
        node_ids = [node["id"] for node in graph["nodes"]]

        for i, node_a in enumerate(node_ids):
            xa, ya = positions[node_a]
            for node_b in node_ids[i + 1 :]:
                xb, yb = positions[node_b]
                # Treat near-identical placement as overlap risk in the editor.
                assert not (abs(xa - xb) < 140.0 and abs(ya - yb) < 90.0)
