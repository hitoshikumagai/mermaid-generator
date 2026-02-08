import pytest

from src.mermaid_generator.editor_policy import (
    get_editor_capabilities,
    get_export_filename,
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
