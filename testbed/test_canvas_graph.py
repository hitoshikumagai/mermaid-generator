from src.mermaid_generator.canvas_graph import graph_to_mermaid, parse_mermaid_to_graph
from src.mermaid_generator.templates import get_mermaid_template, list_mermaid_templates


def _first_template_code(diagram_type: str) -> str:
    template = list_mermaid_templates(diagram_type)[0]
    return get_mermaid_template(diagram_type, template["id"])


def test_parse_mermaid_to_graph_produces_canvas_graph_for_all_non_flowchart_types():
    for diagram_type in ("Sequence", "State", "ER", "Class", "Gantt"):
        graph = parse_mermaid_to_graph(diagram_type, _first_template_code(diagram_type))
        assert graph["nodes"]
        assert graph["edges"] or diagram_type == "Gantt"
        node_ids = {node["id"] for node in graph["nodes"]}
        for edge in graph["edges"]:
            assert edge["source"] in node_ids
            assert edge["target"] in node_ids


def test_graph_to_mermaid_emits_expected_header_for_each_type():
    expected_headers = {
        "Sequence": "sequenceDiagram",
        "State": "stateDiagram-v2",
        "ER": "erDiagram",
        "Class": "classDiagram",
        "Gantt": "gantt",
    }
    for diagram_type, header in expected_headers.items():
        graph = parse_mermaid_to_graph(diagram_type, _first_template_code(diagram_type))
        code = graph_to_mermaid(diagram_type, graph)
        assert code.startswith(header + "\n")


def test_sequence_parser_extracts_participants_and_messages():
    code = (
        "sequenceDiagram\n"
        "    participant A as App\n"
        "    participant B as Backend\n"
        "    A->>B: request\n"
        "    B-->>A: response\n"
    )
    graph = parse_mermaid_to_graph("Sequence", code)
    ids = {node["id"] for node in graph["nodes"]}
    assert ids == {"A", "B"}
    assert len(graph["edges"]) == 2
