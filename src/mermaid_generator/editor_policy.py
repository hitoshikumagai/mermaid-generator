from typing import Dict

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
