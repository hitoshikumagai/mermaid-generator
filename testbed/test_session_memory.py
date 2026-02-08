from src.mermaid_generator.session_memory import (
    append_template_memory,
    build_memory_context,
    should_reset_conversation,
)


def test_append_template_memory_and_build_context():
    store = {}
    append_template_memory(
        store,
        diagram_type="Flowchart",
        template_id="approval",
        template_name="Approval Workflow",
        template_description="Request review and decision flow.",
        bootstrap=False,
    )
    append_template_memory(
        store,
        diagram_type="Flowchart",
        template_id="incident",
        template_name="Incident Response",
        template_description="Detect, triage, mitigate.",
        bootstrap=True,
    )
    context = build_memory_context(store, "Flowchart", max_events=2)

    assert "Approval Workflow" in context
    assert "Incident Response" in context
    assert "stage=interactive" in context
    assert "stage=bootstrap" in context


def test_should_reset_conversation_respects_bootstrap_flag():
    assert should_reset_conversation(bootstrap=False) is True
    assert should_reset_conversation(bootstrap=True) is False
