from src.mermaid_generator.graph_logic import build_mock_graph
from src.mermaid_generator.orchestrator import FlowchartOrchestrator, compute_impact_range


class DisabledClient:
    def is_enabled(self) -> bool:
        return False


class EnabledStubClient:
    def is_enabled(self) -> bool:
        return True

    def complete_json(self, system_prompt, user_prompt, temperature=0.2):
        if "ready_for_diagram" in system_prompt:
            return {
                "assistant_message": "Scope is clear.",
                "scope_summary": "checkout flow",
                "ready_for_diagram": True,
                "graph_brief": "checkout flow",
            }
        if "update an existing flowchart" in system_prompt:
            return {
                "assistant_message": "updated",
                "scope_summary": "checkout flow + refund",
                "impact_message": "refund branch added",
                "graph": {
                    "nodes": [
                        {"id": "start", "label": "Start", "type": "input"},
                        {"id": "refund", "label": "Refund", "type": "default"},
                    ],
                    "edges": [{"id": "e1", "source": "start", "target": "refund", "label": ""}],
                },
            }
        return {
            "nodes": [
                {"id": "start", "label": "Start", "type": "input"},
                {"id": "end", "label": "End", "type": "output"},
            ],
            "edges": [{"id": "e1", "source": "start", "target": "end", "label": ""}],
        }


def test_compute_impact_range_detects_changes():
    prev_graph = {
        "nodes": [
            {"id": "a", "label": "A", "type": "input"},
            {"id": "b", "label": "B", "type": "default"},
        ],
        "edges": [{"id": "e1", "source": "a", "target": "b", "label": ""}],
    }
    next_graph = {
        "nodes": [
            {"id": "a", "label": "A changed", "type": "input"},
            {"id": "c", "label": "C", "type": "output"},
        ],
        "edges": [{"id": "e2", "source": "a", "target": "c", "label": "new"}],
    }

    impact = compute_impact_range(prev_graph, next_graph)

    assert impact["changed_node_ids"] == ["a"]
    assert impact["added_node_ids"] == ["c"]
    assert impact["removed_node_ids"] == ["b"]
    assert impact["added_edge_ids"] == ["e2"]
    assert impact["removed_edge_ids"] == ["e1"]


def test_fallback_first_turn_is_scope_when_too_short():
    orchestrator = FlowchartOrchestrator(llm_client=DisabledClient())
    turn = orchestrator.run_turn("short", [], "", None)

    assert turn.mode == "scope"
    assert turn.graph_data is None
    assert turn.impact["phase"] == "initial"


def test_fallback_second_turn_reports_impact():
    orchestrator = FlowchartOrchestrator(llm_client=DisabledClient())
    current_graph = build_mock_graph("first")

    turn = orchestrator.run_turn(
        user_message="add refund path",
        chat_history=[{"role": "user", "content": "first"}],
        current_scope="first scope",
        current_graph=current_graph,
    )

    assert turn.mode == "visualize"
    assert turn.graph_data is not None
    assert turn.impact["phase"] == "update"
    assert "impacted_node_ids" in turn.impact


def test_llm_second_turn_uses_update_path():
    orchestrator = FlowchartOrchestrator(llm_client=EnabledStubClient())
    current_graph = {
        "nodes": [{"id": "start", "label": "Start", "type": "input"}],
        "edges": [],
    }

    turn = orchestrator.run_turn(
        user_message="add refund",
        chat_history=[{"role": "user", "content": "initial"}],
        current_scope="checkout flow",
        current_graph=current_graph,
    )

    assert turn.mode == "visualize"
    assert turn.source == "llm"
    assert turn.impact["phase"] == "update"
    assert "refund" in [n["id"] for n in turn.graph_data["nodes"]]
