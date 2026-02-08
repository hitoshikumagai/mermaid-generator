from src.mermaid_generator.diagram_management_assistant import DiagramDecisionAssistant


class DisabledClient:
    def is_enabled(self) -> bool:
        return False


class EnabledStubClient:
    def is_enabled(self) -> bool:
        return True

    def complete_json(self, system_prompt, user_prompt, temperature=0.2):
        _ = system_prompt
        _ = user_prompt
        _ = temperature
        return {
            "summary": "Adopted candidate B for review",
            "markdown_comment": (
                "### Decision\n"
                "- compared A and B\n"
                "- selected B due to clearer ownership\n"
            ),
            "tags": ["review", "ownership"],
        }


class ErrorClient:
    def is_enabled(self) -> bool:
        return True

    def complete_json(self, system_prompt, user_prompt, temperature=0.2):
        _ = system_prompt
        _ = user_prompt
        _ = temperature
        raise RuntimeError("network timeout")


def _diagram():
    return {
        "id": "d_001",
        "title": "Approval Workflow",
        "diagram_type": "Flowchart",
        "status": "in_review",
        "mermaid_code": "graph TD; A-->B;",
    }


def test_draft_note_with_llm_returns_llm_payload():
    assistant = DiagramDecisionAssistant(llm_client=EnabledStubClient())
    result = assistant.draft_note(
        diagram=_diagram(),
        user_request="Summarize why candidate B is safer.",
        recent_events=[],
    )

    assert result["source"] == "llm"
    assert "candidate B" in result["summary"]
    assert "selected B" in result["markdown_comment"]
    assert "review" in result["tags"]


def test_draft_note_without_api_key_falls_back_to_markdown():
    assistant = DiagramDecisionAssistant(llm_client=DisabledClient())
    result = assistant.draft_note(
        diagram=_diagram(),
        user_request="Summarize scope tradeoffs.",
        recent_events=[],
    )

    assert result["source"] == "fallback"
    assert "LLM fallback note" in result["markdown_comment"]
    assert "scope tradeoffs" in result["markdown_comment"]
    assert "fallback" in result["tags"]


def test_draft_note_with_api_error_falls_back_to_markdown():
    assistant = DiagramDecisionAssistant(llm_client=ErrorClient())
    result = assistant.draft_note(
        diagram=_diagram(),
        user_request="Document decision impact.",
        recent_events=[],
    )

    assert result["source"] == "fallback"
    assert "network timeout" in result["markdown_comment"]
    assert "decision impact" in result["markdown_comment"]
