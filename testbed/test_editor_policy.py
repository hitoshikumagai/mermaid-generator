import pytest

from src.mermaid_generator.editor_policy import (
    get_editor_capabilities,
    get_export_filename,
    get_focus_layout_policy,
    normalize_editor_mode,
)
from src.mermaid_generator.templates import DIAGRAM_TYPES


def test_all_diagram_types_support_edit_preview_export_in_all_modes():
    for diagram_type in DIAGRAM_TYPES:
        manual = get_editor_capabilities(diagram_type, "Manual")
        orchestration = get_editor_capabilities(diagram_type, "Orchestration")
        assert manual["edit"] and manual["preview"] and manual["export"]
        assert orchestration["edit"] and orchestration["preview"] and orchestration["export"]


def test_chat_capability_is_mode_based():
    capabilities = get_editor_capabilities("Sequence", "Orchestration")
    assert capabilities["chat"] is True

    capabilities = get_editor_capabilities("Sequence", "Manual")
    assert capabilities["chat"] is False


def test_export_filename_extension_is_mmd():
    assert get_export_filename("Flowchart") == "flowchart_diagram.mmd"
    assert get_export_filename("State") == "state_diagram.mmd"


def test_unknown_diagram_type_raises():
    with pytest.raises(ValueError):
        get_editor_capabilities("Unknown", "Manual")


def test_unknown_mode_falls_back_to_manual():
    assert normalize_editor_mode("unexpected") == "Manual"


def test_focus_layout_policy_flowchart_orchestration_prioritizes_chat_and_preview():
    policy = get_focus_layout_policy("Flowchart", "Orchestration")
    assert policy["primary_sections"] == ["chat", "preview"]
    assert policy["chat_enabled"] is True
    assert "canvas_editor" in policy["collapsed_sections"]
    assert "export" in policy["collapsed_sections"]
    assert "candidate_management" in policy["collapsed_sections"]
    assert "impact_debug" in policy["collapsed_sections"]
    assert "property_editor" not in policy["collapsed_sections"]


def test_focus_layout_policy_sequence_manual_collapses_secondary_sections():
    policy = get_focus_layout_policy("Sequence", "Manual")
    assert policy["primary_sections"] == ["preview"]
    assert policy["chat_enabled"] is False
    assert "canvas_editor" in policy["collapsed_sections"]
    assert "export" in policy["collapsed_sections"]
    assert "candidate_management" in policy["collapsed_sections"]
    assert "property_editor" in policy["collapsed_sections"]
    assert "agent_details" in policy["collapsed_sections"]
    assert "impact_debug" not in policy["collapsed_sections"]
