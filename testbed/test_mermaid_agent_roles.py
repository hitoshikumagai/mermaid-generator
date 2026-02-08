from src.mermaid_generator.mermaid_agent_roles import (
    ArchitectRoleAgent,
    InterpreterRoleAgent,
    MermaidActAgent,
    MermaidObserveAgent,
    MermaidRoleCoordinator,
    MermaidVerifyAgent,
    SystemEngineerRoleAgent,
    extract_steps,
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


def test_verify_output_rejects_header_only_candidate():
    verifier = MermaidVerifyAgent(sanitize_mermaid_code)
    model, reason = verifier.verify_output(
        diagram_type="Sequence",
        candidate_code="sequenceDiagram\n",
        fallback_code="",
    )

    assert model is None
    assert reason == "missing body"


def test_verify_output_rejects_missing_type_specific_syntax():
    verifier = MermaidVerifyAgent(sanitize_mermaid_code)
    model, reason = verifier.verify_output(
        diagram_type="Sequence",
        candidate_code="sequenceDiagram\n    this is invalid\n",
        fallback_code="",
    )

    assert model is None
    assert reason == "missing type-specific syntax"


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


def test_coordinator_enforces_final_mermaid_output_model():
    class BrokenEngineer(SystemEngineerRoleAgent):
        def react(self, observation, default_template_code):  # noqa: ANN001
            _ = observation
            _ = default_template_code
            return type(
                "BrokenArtifact",
                (),
                {
                    "mermaid_code": "sequenceDiagram\n",
                    "validation_hints": [],
                    "applied_steps": 1,
                    "validate": lambda self: (True, "ok"),
                },
            )()

        def verify(self, diagram_type, artifact, current_code, user_message):  # noqa: ANN001
            _ = diagram_type
            _ = artifact
            _ = current_code
            _ = user_message
            return True, "ok"

    coordinator = MermaidRoleCoordinator(
        sanitize_mermaid_code,
        interpreter=InterpreterRoleAgent(),
        architect=ArchitectRoleAgent(),
        system_engineer=BrokenEngineer(sanitize_mermaid_code),
    )
    result = coordinator.run(
        diagram_type="Sequence",
        user_message="add triage and archive",
        chat_history=[],
        current_code="sequenceDiagram\n    participant U as User\n",
        default_template_code="sequenceDiagram\n    participant U as User\n",
    )

    assert result.source == "fallback"
    assert "final_output" in result.failed_role
    assert "participant U as User" in result.mermaid_code


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


def test_extract_steps_ignores_session_memory_block():
    steps = extract_steps(
        "1. triage inbox\n"
        "2. quick reply\n\n"
        "Session Memory:\n"
        "- template=Incident Response (id=incident, stage=interactive)\n"
    )

    assert len(steps) >= 2
    assert all("template=" not in step for step in steps)
    assert all("Session Memory" not in step for step in steps)
