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


def test_graph_to_mermaid_flowchart_uses_graph_header():
    graph = {
        "nodes": [
            {"id": "start", "label": "Start", "type": "input"},
            {"id": "end", "label": "End", "type": "output"},
        ],
        "edges": [{"id": "e1", "source": "start", "target": "end", "label": ""}],
    }
    code = graph_to_mermaid("Flowchart", graph)

    assert code.startswith("graph TD;\n")
    assert "start --> end;" in code


def test_sequence_roundtrip_preserves_return_message_direction():
    source = (
        "sequenceDiagram\n"
        "    participant C as Client\n"
        "    participant A as API\n"
        "    C->>A: request\n"
        "    A-->>C: response\n"
    )
    graph = parse_mermaid_to_graph("Sequence", source)
    code = graph_to_mermaid("Sequence", graph)

    assert "C->>A: request" in code
    assert "A-->>C: response" in code


def test_class_roundtrip_preserves_inheritance_relation():
    source = (
        "classDiagram\n"
        "    class Parent\n"
        "    class Child\n"
        "    Parent <|-- Child\n"
    )
    graph = parse_mermaid_to_graph("Class", source)
    code = graph_to_mermaid("Class", graph)

    assert "Child --|> Parent" in code


def test_er_roundtrip_preserves_cardinality_marker():
    source = (
        "erDiagram\n"
        "    USER ||--|{ ORDER : places\n"
    )
    graph = parse_mermaid_to_graph("ER", source)
    code = graph_to_mermaid("ER", graph)

    assert "USER ||--|{ ORDER : places" in code
