from src.mermaid_generator.mermaid_agent_roles import (
    ArchitectRoleAgent,
    InterpreterRoleAgent,
    MermaidActAgent,
    MermaidObserveAgent,
    MermaidRoleCoordinator,
    MermaidVerifyAgent,
    SystemEngineerRoleAgent,
)
from src.mermaid_generator.orchestrator import sanitize_mermaid_code


def test_observe_extracts_steps_from_prose():
    observer = MermaidObserveAgent()
    result = observer.observe(
        user_message="1. triage mail。2. quick reply。3. archive.",
        current_code="sequenceDiagram\n    participant U\n",
    )

    assert result.has_current_code is True
    assert len(result.steps) >= 3
    assert "triage mail" in result.steps[0]


def test_verify_detects_no_effective_update():
    verifier = MermaidVerifyAgent(sanitize_mermaid_code)
    current = "sequenceDiagram\n    participant U as User\n"
    ok, reason = verifier.verify(
        diagram_type="Sequence",
        candidate_code=current,
        current_code=current,
        user_message="add step",
    )

    assert ok is False
    assert reason == "no effective update"


def test_act_generates_fallback_update_from_request():
    actor = MermaidActAgent(sanitize_mermaid_code)
    code = actor.fallback_code(
        diagram_type="Sequence",
        user_message="受信確認。2分以内なら返信。終わったらアーカイブ。",
        current_code="sequenceDiagram\n    participant U as User\n",
        append=True,
        default_template_code="sequenceDiagram\n    participant U as User\n",
    )

    assert "sequenceDiagram" in code
    assert code.count("->>") >= 2


def test_three_role_coordinator_success_path():
    coordinator = MermaidRoleCoordinator(sanitize_mermaid_code)
    result = coordinator.run(
        diagram_type="Sequence",
        user_message="受信確認。2分以内なら返信。完了後にアーカイブ。",
        chat_history=[],
        current_code="sequenceDiagram\n    participant U as User\n",
        default_template_code="sequenceDiagram\n    participant U as User\n",
    )

    assert result.source == "roles"
    assert result.failed_role == ""
    assert result.mermaid_code.count("->>") >= 2


def test_coordinator_falls_back_when_role_verify_fails():
    class BrokenArchitect(ArchitectRoleAgent):
        def verify(self, artifact):  # noqa: ANN001
            _ = artifact
            return False, "forced-failure"

    coordinator = MermaidRoleCoordinator(
        sanitize_mermaid_code,
        interpreter=InterpreterRoleAgent(),
        architect=BrokenArchitect(),
        system_engineer=SystemEngineerRoleAgent(sanitize_mermaid_code),
    )
    result = coordinator.run(
        diagram_type="Sequence",
        user_message="add triage and archive",
        chat_history=[],
        current_code="sequenceDiagram\n    participant U as User\n",
        default_template_code="sequenceDiagram\n    participant U as User\n",
    )

    assert result.source == "fallback"
    assert "architect" in result.failed_role
    assert "sequenceDiagram" in result.mermaid_code


def test_interpreter_architect_reflect_session_memory():
    interpreter = InterpreterRoleAgent()
    obs = interpreter.observe(
        user_message=(
            "add retry and archive\n\n"
            "Session Memory:\n"
            "- template=Incident Response (id=incident, stage=interactive) | Detect and triage flow\n"
        ),
        chat_history=[],
    )
    artifact = interpreter.react(obs)
    assert artifact.memory_lines
    assert "template=Incident Response" in artifact.memory_lines[0]

    architect = ArchitectRoleAgent()
    arch_obs = architect.observe(
        diagram_type="Sequence",
        current_code="sequenceDiagram\n    participant U as User\n",
        interpreter=artifact,
    )
    arch_artifact = architect.react(arch_obs)
    assert "respect-template-memory" in arch_artifact.invariants
