from typing import Dict


def build_validation_feedback(validation: Dict[str, str]) -> Dict[str, str]:
    if not isinstance(validation, dict) or not validation:
        return {"level": "none", "title": "", "message": "", "guidance": ""}

    status = str(validation.get("status", "")).strip().lower()
    reason = str(validation.get("reason", "")).strip()

    if status == "pass":
        return {
            "level": "success",
            "title": "Validation Passed",
            "message": "Diagram validation passed.",
            "guidance": "",
        }

    if status == "recovered":
        return {
            "level": "warning",
            "title": "Validation Recovered",
            "message": (
                "Validation found issues and auto-recovered via fallback generation."
                + (f" Reason: {reason}" if reason else "")
            ),
            "guidance": "Review preview, then refine prompt or edit nodes/edges if needed.",
        }

    if status == "blocked":
        return {
            "level": "error",
            "title": "Validation Blocked Completion",
            "message": (
                "Validation blocked completion."
                + (f" Reason: {reason}" if reason else "")
            ),
            "guidance": "Adjust prompt content or switch to the correct diagram type, then retry.",
        }

    return {
        "level": "info",
        "title": "Validation Status",
        "message": f"Validation status: {status}",
        "guidance": "",
    }
