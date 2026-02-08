import json
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DiagramRecord = Dict[str, Any]
DecisionEvent = Dict[str, Any]
VectorRecord = Dict[str, Any]

ALLOWED_STATUS_TRANSITIONS = {
    "active": {"in_review", "archived"},
    "in_review": {"approved", "active", "archived"},
    "approved": {"archived"},
    "archived": {"active"},
}


class DiagramRepository:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._diagram_file = self.base_dir / "diagrams.json"
        self._event_file = self.base_dir / "decision_events.jsonl"
        self._vector_file = self.base_dir / "vector_records.jsonl"
        if not self._diagram_file.exists():
            self._write_json(self._diagram_file, [])

    def create_diagram(
        self,
        title: str,
        diagram_type: str,
        mermaid_code: str,
        graph_data: Dict[str, Any],
        actor: str = "user",
        tags: Optional[List[str]] = None,
        scope_summary: str = "",
        chat_history: Optional[List[Dict[str, str]]] = None,
        mode: str = "Manual",
    ) -> DiagramRecord:
        diagrams = self._read_diagrams()
        now = _now_utc_iso()
        record = {
            "id": _new_id("d"),
            "title": title.strip() or f"{diagram_type} Candidate",
            "diagram_type": diagram_type,
            "status": "active",
            "mermaid_code": mermaid_code,
            "graph_data": deepcopy(graph_data),
            "tags": list(tags or []),
            "scope_summary": str(scope_summary or ""),
            "chat_history": deepcopy(chat_history or []),
            "mode": str(mode or "Manual"),
            "created_at": now,
            "updated_at": now,
        }
        diagrams.append(record)
        self._write_diagrams(diagrams)
        self.append_decision_event(
            diagram_id=record["id"],
            actor=actor,
            stage="create",
            summary="Diagram created",
            markdown_comment=f"- title: {record['title']}\n- type: {diagram_type}",
            tags=["create"],
        )
        return deepcopy(record)

    def list_diagrams(self, include_archived: bool = False) -> List[DiagramRecord]:
        diagrams = self._read_diagrams()
        if not include_archived:
            diagrams = [d for d in diagrams if d.get("status") != "archived"]
        diagrams.sort(key=lambda d: d.get("updated_at", ""), reverse=True)
        return deepcopy(diagrams)

    def get_diagram(self, diagram_id: str) -> Optional[DiagramRecord]:
        for diagram in self._read_diagrams():
            if diagram.get("id") == diagram_id:
                return deepcopy(diagram)
        return None

    def update_diagram_content(
        self,
        diagram_id: str,
        mermaid_code: str,
        graph_data: Dict[str, Any],
        actor: str = "user",
        scope_summary: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        mode: Optional[str] = None,
    ) -> DiagramRecord:
        diagrams = self._read_diagrams()
        target = _find_diagram(diagrams, diagram_id)
        target["mermaid_code"] = mermaid_code
        target["graph_data"] = deepcopy(graph_data)
        if scope_summary is not None:
            target["scope_summary"] = str(scope_summary)
        if chat_history is not None:
            target["chat_history"] = deepcopy(chat_history)
        if mode is not None:
            target["mode"] = str(mode)
        target["updated_at"] = _now_utc_iso()
        self._write_diagrams(diagrams)
        self.append_decision_event(
            diagram_id=diagram_id,
            actor=actor,
            stage="edit",
            summary="Diagram content updated",
            markdown_comment="- content updated from editor",
            tags=["edit"],
        )
        return deepcopy(target)

    def rename_diagram(self, diagram_id: str, title: str, actor: str = "user") -> DiagramRecord:
        diagrams = self._read_diagrams()
        target = _find_diagram(diagrams, diagram_id)
        previous_title = str(target.get("title", ""))
        updated_title = title.strip() or previous_title
        target["title"] = updated_title
        target["updated_at"] = _now_utc_iso()
        self._write_diagrams(diagrams)
        self.append_decision_event(
            diagram_id=diagram_id,
            actor=actor,
            stage="rename",
            summary="Diagram renamed",
            markdown_comment=f"- from: {previous_title}\n- to: {updated_title}",
            tags=["rename"],
        )
        return deepcopy(target)

    def duplicate_diagram(self, diagram_id: str, actor: str = "user") -> DiagramRecord:
        diagrams = self._read_diagrams()
        source = _find_diagram(diagrams, diagram_id)
        now = _now_utc_iso()
        duplicated = deepcopy(source)
        duplicated["id"] = _new_id("d")
        duplicated["title"] = f"{str(source.get('title', 'Diagram')).strip()} (Copy)"
        duplicated["status"] = "active"
        duplicated["created_at"] = now
        duplicated["updated_at"] = now
        diagrams.append(duplicated)
        self._write_diagrams(diagrams)
        self.append_decision_event(
            diagram_id=duplicated["id"],
            actor=actor,
            stage="duplicate",
            summary="Diagram duplicated",
            markdown_comment=f"- source: {diagram_id}",
            tags=["duplicate"],
        )
        return deepcopy(duplicated)

    def delete_diagram(self, diagram_id: str, actor: str = "user") -> DiagramRecord:
        diagrams = self._read_diagrams()
        target = _find_diagram(diagrams, diagram_id)
        next_diagrams = [diagram for diagram in diagrams if diagram.get("id") != diagram_id]
        self._write_diagrams(next_diagrams)
        deleted_event = {
            "id": _new_id("ev"),
            "diagram_id": diagram_id,
            "actor": actor,
            "stage": "delete",
            "summary": "Diagram deleted",
            "markdown_comment": f"- title: {target.get('title', '')}",
            "tags": ["delete"],
            "created_at": _now_utc_iso(),
        }
        _append_jsonl(self._event_file, deleted_event)
        return deepcopy(target)

    def set_status(self, diagram_id: str, status: str, actor: str = "user", reason: str = "") -> DiagramRecord:
        diagrams = self._read_diagrams()
        target = _find_diagram(diagrams, diagram_id)
        current_status = str(target.get("status", "active"))
        next_status = status.strip()
        allowed = ALLOWED_STATUS_TRANSITIONS.get(current_status, set())
        if next_status not in allowed:
            raise ValueError(f"Invalid status transition: {current_status} -> {next_status}")
        target["status"] = next_status
        target["updated_at"] = _now_utc_iso()
        self._write_diagrams(diagrams)
        self.append_decision_event(
            diagram_id=diagram_id,
            actor=actor,
            stage="status",
            summary=f"Status changed to {next_status}",
            markdown_comment=f"- from: {current_status}\n- to: {next_status}\n- reason: {reason or '(none)'}",
            tags=["status", next_status],
        )
        return deepcopy(target)

    def append_decision_event(
        self,
        diagram_id: str,
        actor: str,
        stage: str,
        summary: str,
        markdown_comment: str,
        tags: Optional[List[str]] = None,
    ) -> DecisionEvent:
        diagram = self.get_diagram(diagram_id)
        if not diagram:
            raise ValueError(f"Diagram not found: {diagram_id}")

        now = _now_utc_iso()
        event = {
            "id": _new_id("ev"),
            "diagram_id": diagram_id,
            "actor": actor,
            "stage": stage,
            "summary": summary.strip(),
            "markdown_comment": markdown_comment.strip(),
            "tags": list(tags or []),
            "created_at": now,
        }
        _append_jsonl(self._event_file, event)

        vector_record = {
            "id": _new_id("vec"),
            "diagram_id": diagram_id,
            "source_event_id": event["id"],
            "text": _build_vector_text(diagram, event),
            "embedding_metadata": {
                "diagram_id": diagram_id,
                "diagram_type": diagram.get("diagram_type", ""),
                "stage": stage,
                "actor": actor,
                "status": diagram.get("status", ""),
                "tags": list(tags or []),
                "created_at": now,
            },
        }
        _append_jsonl(self._vector_file, vector_record)
        return deepcopy(event)

    def list_decision_events(self, diagram_id: str) -> List[DecisionEvent]:
        events = [e for e in _read_jsonl(self._event_file) if e.get("diagram_id") == diagram_id]
        events.sort(key=lambda e: e.get("created_at", ""))
        return deepcopy(events)

    def list_vector_records(self, diagram_id: Optional[str] = None) -> List[VectorRecord]:
        vectors = _read_jsonl(self._vector_file)
        if diagram_id:
            vectors = [v for v in vectors if v.get("diagram_id") == diagram_id]
        vectors.sort(key=lambda v: v.get("id", ""))
        return deepcopy(vectors)

    def _read_diagrams(self) -> List[DiagramRecord]:
        data = self._read_json(self._diagram_file, [])
        return data if isinstance(data, list) else []

    def _write_diagrams(self, diagrams: List[DiagramRecord]) -> None:
        self._write_json(self._diagram_file, diagrams)

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return deepcopy(default)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_json(self, path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _find_diagram(diagrams: List[DiagramRecord], diagram_id: str) -> DiagramRecord:
    for diagram in diagrams:
        if diagram.get("id") == diagram_id:
            return diagram
    raise ValueError(f"Diagram not found: {diagram_id}")


def _build_vector_text(diagram: DiagramRecord, event: DecisionEvent) -> str:
    return (
        f"# Diagram Decision\n"
        f"- Diagram ID: {diagram.get('id')}\n"
        f"- Title: {diagram.get('title')}\n"
        f"- Type: {diagram.get('diagram_type')}\n"
        f"- Status: {diagram.get('status')}\n"
        f"- Stage: {event.get('stage')}\n"
        f"- Summary: {event.get('summary')}\n\n"
        f"{event.get('markdown_comment', '').strip()}\n"
    ).strip()
