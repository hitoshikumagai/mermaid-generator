from src.mermaid_generator.diagram_validator import MermaidDiagramValidator


def test_sequence_validator_detects_unknown_participant():
    validator = MermaidDiagramValidator(enable_llm_assist=False)
    report = validator.validate_turn(
        diagram_type="Sequence",
        candidate_code=(
            "sequenceDiagram\n"
            "    participant U as User\n"
            "    U->>S: request\n"
        ),
        current_code="",
        user_message="add request flow",
    )

    assert report.valid is False
    assert any(item.rule_id == "unknown_participant" for item in report.findings)


def test_gantt_validator_warns_for_unknown_dependency():
    validator = MermaidDiagramValidator(enable_llm_assist=False)
    report = validator.validate_turn(
        diagram_type="Gantt",
        candidate_code=(
            "gantt\n"
            "    title Sample\n"
            "    section Plan\n"
            "    task A :a1, 2026-02-01, 1d\n"
            "    task B :after missing_task, 1d\n"
        ),
        current_code="",
        user_message="build gantt",
    )

    assert report.valid is True
    assert any(item.rule_id == "unknown_dependency_task" for item in report.findings)
