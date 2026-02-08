from datetime import datetime, timezone
from typing import Dict, List

MemoryEvent = Dict[str, str]
MemoryStore = Dict[str, List[MemoryEvent]]


def append_template_memory(
    memory_store: MemoryStore,
    diagram_type: str,
    template_id: str,
    template_name: str,
    template_description: str,
    bootstrap: bool = False,
) -> None:
    events = memory_store.setdefault(diagram_type, [])
    events.append(
        {
            "kind": "template_load",
            "diagram_type": diagram_type,
            "template_id": template_id,
            "template_name": template_name,
            "template_description": template_description.strip(),
            "stage": "bootstrap" if bootstrap else "interactive",
            "timestamp": _now_utc_iso(),
        }
    )


def build_memory_context(memory_store: MemoryStore, diagram_type: str, max_events: int = 3) -> str:
    events = memory_store.get(diagram_type, [])
    if not events:
        return ""
    lines = []
    for event in events[-max_events:]:
        if event.get("kind") != "template_load":
            continue
        line = (
            f"- template={event.get('template_name', '')} "
            f"(id={event.get('template_id', '')}, stage={event.get('stage', '')})"
        ).strip()
        desc = event.get("template_description", "")
        if desc:
            line += f" | {desc}"
        lines.append(line)
    return "\n".join(lines)


def should_reset_conversation(bootstrap: bool) -> bool:
    return not bootstrap


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
