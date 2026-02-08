from src.mermaid_generator.parent_class_assistant import ParentClassAssistant


class DisabledClient:
    def is_enabled(self) -> bool:
        return False


class EnabledStubClient:
    def is_enabled(self) -> bool:
        return True

    def complete_json(self, system_prompt, user_prompt, temperature=0.1):
        _ = system_prompt
        _ = user_prompt
        _ = temperature
        return {
            "parent_id": "Entity",
            "reason": "OrderService should inherit shared fields and contracts from Entity.",
        }


def _class_graph():
    return {
        "nodes": [
            {"id": "Entity", "label": "Entity", "type": "default"},
            {"id": "OrderService", "label": "OrderService", "type": "default"},
            {"id": "PaymentService", "label": "PaymentService", "type": "default"},
        ],
        "edges": [],
    }


def test_suggest_parent_with_llm_response():
    assistant = ParentClassAssistant(llm_client=EnabledStubClient())
    result = assistant.suggest_parent(
        graph_data=_class_graph(),
        child_id="OrderService",
        user_request="Suggest a reusable base class.",
    )

    assert result["source"] == "llm"
    assert result["child_id"] == "OrderService"
    assert result["parent_id"] == "Entity"
    assert "inherit" in result["reason"]


def test_suggest_parent_falls_back_when_llm_disabled():
    assistant = ParentClassAssistant(llm_client=DisabledClient())
    result = assistant.suggest_parent(
        graph_data=_class_graph(),
        child_id="OrderService",
        user_request="Prefer domain base class.",
    )

    assert result["source"] == "fallback"
    assert result["child_id"] == "OrderService"
    assert result["parent_id"] in {"Entity", "PaymentService"}
    assert "fallback" in result["reason"].lower()
