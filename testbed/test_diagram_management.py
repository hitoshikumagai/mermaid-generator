import pytest

from src.mermaid_generator.diagram_management import DiagramRepository


def _sample_graph():
    return {
        "nodes": [
            {"id": "start", "label": "Start", "type": "input"},
            {"id": "end", "label": "End", "type": "output"},
        ],
        "edges": [{"id": "e1", "source": "start", "target": "end", "label": ""}],
    }


def test_create_list_update_archive_diagram(tmp_path):
    repo = DiagramRepository(tmp_path)
    created = repo.create_diagram(
        title="Candidate A",
        diagram_type="Flowchart",
        mermaid_code="graph TD;\nstart-->end;\n",
        graph_data=_sample_graph(),
        actor="tester",
    )

    listed = repo.list_diagrams()
    assert len(listed) == 1
    assert listed[0]["id"] == created["id"]
    assert listed[0]["status"] == "active"

    updated = repo.update_diagram_content(
        created["id"],
        mermaid_code="graph TD;\nstart-->middle;\nmiddle-->end;\n",
        graph_data=_sample_graph(),
        actor="tester",
    )
    assert "middle" in updated["mermaid_code"]

    archived = repo.set_status(created["id"], "archived", actor="tester", reason="obsolete")
    assert archived["status"] == "archived"
    assert repo.list_diagrams() == []
    assert len(repo.list_diagrams(include_archived=True)) == 1


def test_decision_history_is_saved_and_queryable(tmp_path):
    repo = DiagramRepository(tmp_path)
    diagram = repo.create_diagram(
        title="Candidate B",
        diagram_type="Class",
        mermaid_code="classDiagram\nclass A\nclass B\nA --|> B\n",
        graph_data=_sample_graph(),
        actor="tester",
    )

    event = repo.append_decision_event(
        diagram_id=diagram["id"],
        actor="tester",
        stage="review",
        summary="Choose inheritance model",
        markdown_comment="- considered option A\n- selected option B",
        tags=["architecture", "inheritance"],
    )
    assert event["diagram_id"] == diagram["id"]
    assert event["summary"] == "Choose inheritance model"

    events = repo.list_decision_events(diagram["id"])
    assert len(events) == 2  # includes creation event
    assert events[-1]["summary"] == "Choose inheritance model"
    assert "inheritance" in events[-1]["tags"]

    vectors = repo.list_vector_records(diagram["id"])
    assert vectors
    assert vectors[-1]["diagram_id"] == diagram["id"]
    assert vectors[-1]["embedding_metadata"]["stage"] == "review"
    assert "selected option B" in vectors[-1]["text"]


def test_status_transition_rules_are_enforced(tmp_path):
    repo = DiagramRepository(tmp_path)
    diagram = repo.create_diagram(
        title="Candidate C",
        diagram_type="Sequence",
        mermaid_code="sequenceDiagram\nA->>B: ping\n",
        graph_data=_sample_graph(),
        actor="tester",
    )

    repo.set_status(diagram["id"], "in_review", actor="tester", reason="ready")
    repo.set_status(diagram["id"], "approved", actor="tester", reason="accepted")

    with pytest.raises(ValueError):
        repo.set_status(diagram["id"], "draft", actor="tester", reason="rollback")
