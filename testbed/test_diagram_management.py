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
        scope_summary="purchase flow",
        chat_history=[{"role": "user", "content": "initial scope"}],
        mode="Orchestration",
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
        scope_summary="purchase + payment",
        chat_history=[{"role": "assistant", "content": "added payment"}],
        mode="Manual",
    )
    assert "middle" in updated["mermaid_code"]
    assert updated["scope_summary"] == "purchase + payment"
    assert updated["chat_history"][0]["content"] == "added payment"
    assert updated["mode"] == "Manual"

    archived = repo.set_status(created["id"], "archived", actor="tester", reason="obsolete")
    assert archived["status"] == "archived"
    assert repo.list_diagrams() == []
    assert len(repo.list_diagrams(include_archived=True)) == 1


def test_crud_rename_duplicate_delete_and_restore_state(tmp_path):
    repo = DiagramRepository(tmp_path)
    created = repo.create_diagram(
        title="Workspace A",
        diagram_type="Sequence",
        mermaid_code="sequenceDiagram\nA->>B: ping\n",
        graph_data=_sample_graph(),
        actor="tester",
        scope_summary="api handshake",
        chat_history=[{"role": "user", "content": "need retries"}],
        mode="Manual",
    )

    renamed = repo.rename_diagram(created["id"], "Workspace A v2", actor="tester")
    assert renamed["title"] == "Workspace A v2"

    duplicated = repo.duplicate_diagram(created["id"], actor="tester")
    assert duplicated["id"] != created["id"]
    assert duplicated["title"].startswith("Workspace A v2")
    assert duplicated["scope_summary"] == "api handshake"
    assert duplicated["chat_history"][0]["content"] == "need retries"
    assert duplicated["mode"] == "Manual"

    fetched = repo.get_diagram(duplicated["id"])
    assert fetched is not None
    assert fetched["scope_summary"] == "api handshake"
    assert fetched["chat_history"][0]["content"] == "need retries"
    assert fetched["mode"] == "Manual"

    removed = repo.delete_diagram(created["id"], actor="tester")
    assert removed["id"] == created["id"]
    assert repo.get_diagram(created["id"]) is None
    assert repo.get_diagram(duplicated["id"]) is not None


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
    assert all(vector["diagram_id"] == diagram["id"] for vector in vectors)
    review_vectors = [vector for vector in vectors if vector["embedding_metadata"]["stage"] == "review"]
    assert review_vectors
    assert "selected option B" in review_vectors[0]["text"]


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


def test_create_and_update_keep_mutable_inputs_isolated(tmp_path):
    repo = DiagramRepository(tmp_path)
    graph_data = _sample_graph()
    chat_history = [{"role": "user", "content": "initial scope"}]

    created = repo.create_diagram(
        title="Isolation",
        diagram_type="Flowchart",
        mermaid_code="graph TD;\nstart-->end;\n",
        graph_data=graph_data,
        actor="tester",
        scope_summary="scope-v1",
        chat_history=chat_history,
        mode="Manual",
    )

    graph_data["nodes"][0]["label"] = "MUTATED"
    chat_history[0]["content"] = "MUTATED"
    fetched_after_create = repo.get_diagram(created["id"])
    assert fetched_after_create is not None
    assert fetched_after_create["graph_data"]["nodes"][0]["label"] == "Start"
    assert fetched_after_create["chat_history"][0]["content"] == "initial scope"

    repo.update_diagram_content(
        created["id"],
        mermaid_code="graph TD;\nstart-->middle;\nmiddle-->end;\n",
        graph_data=_sample_graph(),
        actor="tester",
    )
    fetched_after_update = repo.get_diagram(created["id"])
    assert fetched_after_update is not None
    assert fetched_after_update["scope_summary"] == "scope-v1"
    assert fetched_after_update["chat_history"][0]["content"] == "initial scope"
    assert fetched_after_update["mode"] == "Manual"


def test_rename_blank_title_keeps_previous_value_and_logs_event(tmp_path):
    repo = DiagramRepository(tmp_path)
    created = repo.create_diagram(
        title="Stable Title",
        diagram_type="Sequence",
        mermaid_code="sequenceDiagram\nA->>B: ping\n",
        graph_data=_sample_graph(),
        actor="tester",
    )

    renamed = repo.rename_diagram(created["id"], "   ", actor="tester")
    assert renamed["title"] == "Stable Title"

    events = repo.list_decision_events(created["id"])
    assert events[-1]["stage"] == "rename"
    assert "- to: Stable Title" in events[-1]["markdown_comment"]


def test_delete_adds_delete_event_without_new_vector_record(tmp_path):
    repo = DiagramRepository(tmp_path)
    created = repo.create_diagram(
        title="Delete Candidate",
        diagram_type="Class",
        mermaid_code="classDiagram\nclass A\n",
        graph_data=_sample_graph(),
        actor="tester",
    )
    vector_count_before = len(repo.list_vector_records(created["id"]))

    deleted = repo.delete_diagram(created["id"], actor="tester")
    assert deleted["id"] == created["id"]
    assert repo.get_diagram(created["id"]) is None

    events = repo.list_decision_events(created["id"])
    assert events[-1]["stage"] == "delete"
    assert events[-1]["summary"] == "Diagram deleted"
    assert len(repo.list_vector_records(created["id"])) == vector_count_before


@pytest.mark.parametrize(
    "operation",
    [
        lambda r: r.update_diagram_content("d_missing", "graph TD;\nA-->B;\n", _sample_graph(), actor="tester"),
        lambda r: r.rename_diagram("d_missing", "new title", actor="tester"),
        lambda r: r.duplicate_diagram("d_missing", actor="tester"),
        lambda r: r.delete_diagram("d_missing", actor="tester"),
    ],
)
def test_missing_diagram_operations_raise_value_error(tmp_path, operation):
    repo = DiagramRepository(tmp_path)
    with pytest.raises(ValueError, match="Diagram not found"):
        operation(repo)


def test_malformed_diagrams_json_is_treated_as_empty(tmp_path):
    (tmp_path / "diagrams.json").write_text("{invalid json", encoding="utf-8")
    repo = DiagramRepository(tmp_path)

    assert repo.list_diagrams() == []


def test_malformed_jsonl_rows_are_skipped_in_event_and_vector_queries(tmp_path):
    repo = DiagramRepository(tmp_path)
    created = repo.create_diagram(
        title="Candidate D",
        diagram_type="Flowchart",
        mermaid_code="graph TD;\nstart-->end;\n",
        graph_data=_sample_graph(),
        actor="tester",
    )

    with (tmp_path / "decision_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not-a-json-row}\n")
    with (tmp_path / "vector_records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not-a-json-row}\n")

    events = repo.list_decision_events(created["id"])
    vectors = repo.list_vector_records(created["id"])
    assert events
    assert vectors
    assert all(event["diagram_id"] == created["id"] for event in events)
    assert all(vector["diagram_id"] == created["id"] for vector in vectors)


def test_mode_whitespace_is_normalized_to_manual_on_create_and_update(tmp_path):
    repo = DiagramRepository(tmp_path)
    created = repo.create_diagram(
        title="Mode Check",
        diagram_type="Flowchart",
        mermaid_code="graph TD;\nstart-->end;\n",
        graph_data=_sample_graph(),
        actor="tester",
        mode="   ",
    )
    assert created["mode"] == "Manual"

    updated = repo.update_diagram_content(
        created["id"],
        mermaid_code="graph TD;\nstart-->middle;\nmiddle-->end;\n",
        graph_data=_sample_graph(),
        actor="tester",
        mode="   ",
    )
    assert updated["mode"] == "Manual"
