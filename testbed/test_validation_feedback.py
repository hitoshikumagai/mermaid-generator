from src.mermaid_generator.validation_feedback import build_validation_feedback


def test_feedback_for_recovered_validation_has_warning_and_guidance():
    feedback = build_validation_feedback({"status": "recovered", "reason": "diagram_type_mismatch"})
    assert feedback["level"] == "warning"
    assert "auto-recovered" in feedback["message"]
    assert feedback["guidance"]


def test_feedback_for_blocked_validation_has_error_and_action():
    feedback = build_validation_feedback({"status": "blocked", "reason": "header mismatch"})
    assert feedback["level"] == "error"
    assert "blocked" in feedback["message"].lower()
    assert "retry" in feedback["guidance"].lower()
