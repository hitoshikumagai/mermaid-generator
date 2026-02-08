from src.mermaid_generator.mermaid_agent_roles import (
    MermaidActAgent,
    MermaidObserveAgent,
    MermaidVerifyAgent,
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
