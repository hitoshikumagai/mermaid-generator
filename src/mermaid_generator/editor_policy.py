from typing import Dict, List

from .templates import DIAGRAM_TYPES

EDITOR_MODES = {"Orchestration", "Manual"}


def normalize_editor_mode(mode: str) -> str:
    if mode in EDITOR_MODES:
        return mode
    return "Manual"


def get_editor_capabilities(diagram_type: str, mode: str) -> Dict[str, bool]:
    if diagram_type not in DIAGRAM_TYPES:
        raise ValueError(f"Unknown diagram type: {diagram_type}")

    normalized_mode = normalize_editor_mode(mode)
    return {
        "edit": True,
        "preview": True,
        "export": True,
        "chat": normalized_mode == "Orchestration",
    }


def get_export_filename(diagram_type: str) -> str:
    base = diagram_type.strip().lower().replace(" ", "_")
    return f"{base}_diagram.mmd"


def get_focus_layout_policy(diagram_type: str, mode: str) -> Dict[str, object]:
    capabilities = get_editor_capabilities(diagram_type, mode)
    primary_sections: List[str] = ["preview"]
    if capabilities["chat"]:
        primary_sections.insert(0, "chat")

    collapsed_sections = [
        "canvas_editor",
        "export",
        "candidate_management",
    ]
    if diagram_type == "Flowchart":
        collapsed_sections.append("impact_debug")
    else:
        collapsed_sections.extend(["property_editor", "agent_details"])

    return {
        "primary_sections": primary_sections,
        "collapsed_sections": collapsed_sections,
        "chat_enabled": capabilities["chat"],
        "preview_enabled": capabilities["preview"],
    }
