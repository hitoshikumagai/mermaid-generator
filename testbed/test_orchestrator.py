from src.mermaid_generator.graph_logic import build_mock_graph
from src.mermaid_generator.diagram_validator import ValidationFinding, ValidationReport
from src.mermaid_generator.orchestrator import (
    FlowchartOrchestrator,
    MermaidDiagramOrchestrator,
    compute_impact_range,
    sanitize_mermaid_code,
)


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


class MermaidEnabledStubClient:
    def is_enabled(self) -> bool:
        return True

    def complete_json(self, system_prompt, user_prompt, temperature=0.2):
        if "update Mermaid diagram code" in system_prompt:
            return {
                "assistant_message": "Updated sequence.",
                "change_summary": "Added payment response path.",
                "mermaid_code": (
                    "sequenceDiagram\n"
                    "    participant C as Client\n"
                    "    participant A as API\n"
                    "    C->>A: request\n"
                    "    A-->>C: response\n"
                ),
            }
        return {
            "assistant_message": "Initial draft ready.",
            "change_summary": "Created first diagram.",
            "mermaid_code": "participant U as User\nU->>S: start",
        }


class MermaidNoChangeStubClient:
    def is_enabled(self) -> bool:
        return True

    def complete_json(self, system_prompt, user_prompt, temperature=0.2):
        _ = system_prompt
        _ = user_prompt
        _ = temperature
        return {
            "assistant_message": "No change.",
            "change_summary": "No visible update.",
            "mermaid_code": "sequenceDiagram\n    participant C as Client\n",
        }


class MermaidRepairStubClient:
    def is_enabled(self) -> bool:
        return True

    def complete_json(self, system_prompt, user_prompt, temperature=0.2):
        _ = user_prompt
        _ = temperature
        if "You fix invalid Mermaid code." in system_prompt:
            return {
                "assistant_message": "Repaired code.",
                "change_summary": "Adjusted syntax.",
                "mermaid_code": (
                    "sequenceDiagram\n"
                    "    participant C as Client\n"
                    "    participant A as API\n"
                    "    C->>A: request\n"
                    "    A-->>C: response\n"
                ),
            }
        return {
            "assistant_message": "Initial draft ready.",
            "change_summary": "Created first diagram.",
            "mermaid_code": "sequenceDiagram\n    invalid line\n",
        }


class CountingValidator:
    def __init__(self):
        self.calls = []

    def validate_turn(self, diagram_type, candidate_code, current_code, user_message):
        self.calls.append(
            {
                "diagram_type": diagram_type,
                "candidate_code": candidate_code,
                "current_code": current_code,
                "user_message": user_message,
            }
        )
        return ValidationReport(source="stub", findings=[])


class FailingThenPassingValidator:
    def __init__(self):
        self.calls = 0

    def validate_turn(self, diagram_type, candidate_code, current_code, user_message):
        _ = diagram_type
        _ = candidate_code
        _ = current_code
        _ = user_message
        self.calls += 1
        if self.calls == 1:
            return ValidationReport(
                source="stub",
                findings=[ValidationFinding("error", "forced_failure", "forced failure")],
            )
        return ValidationReport(source="stub", findings=[])


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


def test_flowchart_strict_llm_blocks_fallback_when_llm_disabled():
    orchestrator = FlowchartOrchestrator(llm_client=DisabledClient())
    turn = orchestrator.run_turn(
        user_message="build workflow",
        chat_history=[],
        current_scope="",
        current_graph=None,
        strict_llm=True,
    )

    assert turn.source == "llm_strict_blocked"
    assert turn.graph_data is None
    assert "LLM-only mode" in turn.assistant_message


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


def test_flowchart_fallback_does_not_put_memory_block_in_label():
    orchestrator = FlowchartOrchestrator(llm_client=DisabledClient())
    turn = orchestrator.run_turn(
        user_message=(
            "メール処理を可視化したい。\n\n"
            "Session Memory:\n"
            "- template=EC Purchase Flow (id=ec_purchase, stage=bootstrap)\n"
        ),
        chat_history=[],
        current_scope="",
        current_graph=None,
    )

    assert turn.graph_data is not None
    labels = [node["label"] for node in turn.graph_data["nodes"]]
    assert all("Session Memory" not in label for label in labels)
    assert all("template=" not in label for label in labels)


def test_flowchart_fallback_expands_multistep_text():
    orchestrator = FlowchartOrchestrator(llm_client=DisabledClient())
    turn = orchestrator.run_turn(
        user_message=(
            "1. Decide schedule\n"
            "Handle inbox in batches.\n"
            "2. Apply two-minute rule\n"
            "Reply quickly when possible.\n"
            "3. Keep folder structure minimal\n"
            "4. Reuse templates\n"
        ),
        chat_history=[],
        current_scope="",
        current_graph=None,
    )

    assert turn.graph_data is not None
    assert len(turn.graph_data["nodes"]) >= 7


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


def test_mermaid_fallback_initial_loads_template():
    orchestrator = MermaidDiagramOrchestrator(llm_client=DisabledClient())
    turn = orchestrator.run_turn(
        diagram_type="Sequence",
        user_message="create api sequence",
        chat_history=[],
        current_code="",
    )

    assert turn.source == "fallback"
    assert turn.phase == "initial"
    assert "sequenceDiagram" in turn.mermaid_code


def test_mermaid_strict_llm_blocks_fallback_when_llm_disabled():
    orchestrator = MermaidDiagramOrchestrator(llm_client=DisabledClient())
    turn = orchestrator.run_turn(
        diagram_type="Sequence",
        user_message="create sequence",
        chat_history=[],
        current_code="",
        strict_llm=True,
    )

    assert turn.source == "llm_strict_blocked"
    assert "sequenceDiagram" in turn.mermaid_code
    assert "LLM-only mode" in turn.assistant_message


def test_mermaid_orchestrator_always_calls_dedicated_validator_before_return():
    validator = CountingValidator()
    orchestrator = MermaidDiagramOrchestrator(
        llm_client=DisabledClient(),
        diagram_validator=validator,
    )
    turn = orchestrator.run_turn(
        diagram_type="Sequence",
        user_message="create api sequence",
        chat_history=[],
        current_code="",
    )

    assert turn.mermaid_code.startswith("sequenceDiagram")
    assert len(validator.calls) == 1


def test_mermaid_orchestrator_recovers_with_fallback_when_dedicated_validator_fails():
    validator = FailingThenPassingValidator()
    orchestrator = MermaidDiagramOrchestrator(
        llm_client=MermaidEnabledStubClient(),
        diagram_validator=validator,
    )
    turn = orchestrator.run_turn(
        diagram_type="Sequence",
        user_message="add response step",
        chat_history=[{"role": "user", "content": "initial"}],
        current_code="sequenceDiagram\n    participant C as Client\n",
    )

    assert validator.calls >= 2
    assert turn.source == "fallback"
    assert turn.mermaid_code.startswith("sequenceDiagram")


def test_mermaid_llm_update_uses_update_path():
    orchestrator = MermaidDiagramOrchestrator(llm_client=MermaidEnabledStubClient())
    turn = orchestrator.run_turn(
        diagram_type="Sequence",
        user_message="add response step",
        chat_history=[{"role": "user", "content": "initial"}],
        current_code="sequenceDiagram\n    participant C as Client\n",
    )

    assert turn.source == "llm"
    assert turn.phase == "update"
    assert "A-->>C: response" in turn.mermaid_code
    assert "payment response path" in turn.change_summary


def test_mermaid_fallback_update_builds_from_prose_instruction():
    orchestrator = MermaidDiagramOrchestrator(llm_client=DisabledClient())
    turn = orchestrator.run_turn(
        diagram_type="Sequence",
        user_message=(
            "メール処理の手順: 受信メールを確認。2分で返信可能なら返信。"
            "難しければタスク化して処理。完了後にアーカイブ。"
        ),
        chat_history=[],
        current_code="sequenceDiagram\n    participant C as Client\n",
    )

    assert turn.source == "fallback"
    assert turn.phase == "update"
    assert "A-->>C: response" not in turn.mermaid_code
    assert turn.mermaid_code.count("->>") >= 2
    assert "メール処理" in turn.mermaid_code


def test_mermaid_llm_no_change_triggers_verify_fallback():
    orchestrator = MermaidDiagramOrchestrator(llm_client=MermaidNoChangeStubClient())
    current_code = "sequenceDiagram\n    participant C as Client\n"
    turn = orchestrator.run_turn(
        diagram_type="Sequence",
        user_message="add triage and archive steps",
        chat_history=[{"role": "user", "content": "initial"}],
        current_code=current_code,
    )

    assert turn.source == "fallback"
    assert turn.phase == "update"
    assert turn.mermaid_code != current_code
    assert "verify" in turn.assistant_message.lower()


def test_mermaid_llm_verify_failure_can_be_repaired_by_llm():
    orchestrator = MermaidDiagramOrchestrator(llm_client=MermaidRepairStubClient())
    turn = orchestrator.run_turn(
        diagram_type="Sequence",
        user_message="create request response flow",
        chat_history=[],
        current_code="",
    )

    assert turn.source == "llm"
    assert "C->>A: request" in turn.mermaid_code
    assert "Auto-repaired" in turn.change_summary


def test_sanitize_mermaid_code_strips_fence_and_adds_header():
    raw = "```mermaid\nparticipant U as User\nU->>S: start\n```"
    code = sanitize_mermaid_code("Sequence", raw)

    assert code.startswith("sequenceDiagram\n")
    assert "```" not in code
