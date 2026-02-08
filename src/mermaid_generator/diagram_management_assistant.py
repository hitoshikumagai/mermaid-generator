from typing import Any, Dict, List, Optional

from .orchestrator import OpenAIJSONClient


class DiagramDecisionAssistant:
    def __init__(self, llm_client: Optional[OpenAIJSONClient] = None) -> None:
        self.llm_client = llm_client or OpenAIJSONClient()

    def draft_note(
        self,
        diagram: Dict[str, Any],
        user_request: str,
        recent_events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        request = str(user_request or "").strip() or "Summarize current decision state."
        if not self.llm_client.is_enabled():
            return self._fallback_note(diagram, request, "OPENAI_API_KEY is not configured.")

        try:
            return self._run_llm_note(diagram, request, recent_events)
        except Exception as exc:
            return self._fallback_note(diagram, request, str(exc))

    def _run_llm_note(
        self,
        diagram: Dict[str, Any],
        request: str,
        recent_events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        diagram_type = str(diagram.get("diagram_type", "")).strip() or "Unknown"
        event_lines = []
        for event in recent_events[-5:]:
            summary = str(event.get("summary", "")).strip()
            if summary:
                event_lines.append(f"- {summary}")
        if not event_lines:
            event_lines.append("- (no prior events)")

        system_prompt = (
            "You are a diagram decision assistant. "
            "Return JSON only with keys: summary, markdown_comment, tags.\n"
            "markdown_comment must be concise markdown bullets that explain rationale and impact."
        )
        user_prompt = (
            f"Diagram ID: {diagram.get('id', '')}\n"
            f"Diagram title: {diagram.get('title', '')}\n"
            f"Diagram type: {diagram_type}\n"
            f"Current status: {diagram.get('status', '')}\n\n"
            "Recent decision events:\n"
            f"{chr(10).join(event_lines)}\n\n"
            "User request:\n"
            f"{request}\n\n"
            "Rules:\n"
            "- summary should be one short sentence.\n"
            "- markdown_comment should include rationale, impact scope, and next action.\n"
            "- tags should be 2 to 5 concise keywords.\n"
        )
        payload = self.llm_client.complete_json(system_prompt, user_prompt, temperature=0.1)
        summary = str(payload.get("summary", "")).strip() or "LLM review note drafted."
        markdown_comment = str(payload.get("markdown_comment", "")).strip()
        if not markdown_comment:
            markdown_comment = self._fallback_markdown(diagram, request, "LLM returned empty markdown_comment.")
        raw_tags = payload.get("tags", [])
        tags: List[str] = []
        if isinstance(raw_tags, list):
            for item in raw_tags:
                cleaned = str(item).strip().lower().replace(" ", "-")
                if cleaned:
                    tags.append(cleaned)
        if not tags:
            tags = ["llm", diagram_type.lower()]
        return {
            "summary": summary,
            "markdown_comment": markdown_comment,
            "tags": tags,
            "source": "llm",
        }

    def _fallback_note(self, diagram: Dict[str, Any], request: str, reason: str) -> Dict[str, Any]:
        diagram_type = str(diagram.get("diagram_type", "")).strip() or "unknown"
        summary = f"Fallback note: {request[:72]}"
        markdown_comment = self._fallback_markdown(diagram, request, reason)
        return {
            "summary": summary,
            "markdown_comment": markdown_comment,
            "tags": ["fallback", "manual-review", diagram_type.lower()],
            "source": "fallback",
        }

    def _fallback_markdown(self, diagram: Dict[str, Any], request: str, reason: str) -> str:
        return (
            "### LLM fallback note\n"
            f"- reason: {reason}\n"
            f"- diagram_id: {diagram.get('id', '')}\n"
            f"- diagram_type: {diagram.get('diagram_type', '')}\n"
            f"- user_request: {request}\n"
            "- action: add manual review points and continue discussion in this note."
        )
