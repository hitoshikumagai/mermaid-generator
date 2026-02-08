import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .graph_logic import GraphData, build_mock_graph
from .mermaid_agent_roles import (
    MermaidActAgent,
    MermaidObserveAgent,
    MermaidRoleCoordinator,
    MermaidVerifyAgent,
)
from .templates import get_mermaid_template, list_mermaid_templates

ChatMessage = Dict[str, str]


@dataclass
class AgentTurn:
    assistant_message: str
    scope_summary: str
    graph_data: Optional[GraphData]
    mode: str
    source: str
    impact: Dict[str, Any]


@dataclass
class MermaidTurn:
    assistant_message: str
    mermaid_code: str
    source: str
    phase: str
    change_summary: str


class OpenAIJSONClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: int = 40,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout_seconds = timeout_seconds

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def complete_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
        if not self.is_enabled():
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url="https://api.openai.com/v1/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error: {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error: {exc}") from exc

        parsed = json.loads(raw)
        content = parsed["choices"][0]["message"]["content"]
        return _extract_json_object(content)


class FlowchartOrchestrator:
    def __init__(self, llm_client: Optional[OpenAIJSONClient] = None) -> None:
        self.llm_client = llm_client or OpenAIJSONClient()

    def run_turn(
        self,
        user_message: str,
        chat_history: List[ChatMessage],
        current_scope: str,
        current_graph: Optional[GraphData],
    ) -> AgentTurn:
        has_graph = bool(current_graph and current_graph.get("nodes"))

        if not self.llm_client.is_enabled():
            if has_graph:
                return self._run_fallback_update(user_message, current_scope, current_graph)
            return self._run_fallback_initial(user_message, current_scope)

        try:
            if has_graph:
                return self._run_llm_update(user_message, chat_history, current_scope, current_graph)
            return self._run_llm_initial(user_message, chat_history, current_scope)
        except Exception as exc:
            if has_graph:
                fallback = self._run_fallback_update(user_message, current_scope, current_graph)
            else:
                fallback = self._run_fallback_initial(user_message, current_scope)
            fallback.assistant_message = (
                f"LLM call failed, switched to fallback mode: {exc}"
            )
            fallback.source = "fallback"
            return fallback

    def _run_llm_initial(
        self,
        user_message: str,
        chat_history: List[ChatMessage],
        current_scope: str,
    ) -> AgentTurn:
        scope_result = self._scope_step(user_message, chat_history, current_scope)
        ready = bool(scope_result.get("ready_for_diagram", False))
        assistant_message = str(scope_result.get("assistant_message", "")).strip()
        scope_summary = str(scope_result.get("scope_summary", "")).strip() or current_scope

        if not assistant_message:
            assistant_message = "Scope updated. Add details if needed."

        if not ready:
            return AgentTurn(
                assistant_message=assistant_message,
                scope_summary=scope_summary,
                graph_data=None,
                mode="scope",
                source="llm",
                impact={
                    "phase": "initial",
                    "message": "Waiting for clearer scope.",
                    "impacted_node_ids": [],
                    "impacted_edge_ids": [],
                },
            )

        graph_prompt = str(scope_result.get("graph_brief", "")).strip() or scope_summary
        graph_data = self._graph_step(graph_prompt)
        return AgentTurn(
            assistant_message=assistant_message,
            scope_summary=scope_summary,
            graph_data=graph_data,
            mode="visualize",
            source="llm",
            impact={
                "phase": "initial",
                "message": "Initial diagram generated.",
                "impacted_node_ids": sorted([node["id"] for node in graph_data["nodes"]]),
                "impacted_edge_ids": sorted([edge["id"] for edge in graph_data["edges"]]),
            },
        )

    def _run_llm_update(
        self,
        user_message: str,
        chat_history: List[ChatMessage],
        current_scope: str,
        current_graph: GraphData,
    ) -> AgentTurn:
        update_result = self._update_step(user_message, chat_history, current_scope, current_graph)
        assistant_message = str(update_result.get("assistant_message", "")).strip()
        scope_summary = str(update_result.get("scope_summary", "")).strip() or current_scope
        graph_data = sanitize_graph_data(update_result.get("graph", {}))
        impact = compute_impact_range(current_graph, graph_data)
        impact["phase"] = "update"
        impact["message"] = str(update_result.get("impact_message", "")).strip() or impact["message"]

        if not assistant_message:
            assistant_message = "Diagram updated with your request."

        return AgentTurn(
            assistant_message=assistant_message,
            scope_summary=scope_summary,
            graph_data=graph_data,
            mode="visualize",
            source="llm",
            impact=impact,
        )

    def _scope_step(
        self, user_message: str, chat_history: List[ChatMessage], current_scope: str
    ) -> Dict[str, Any]:
        history_text = _format_history(chat_history[-8:])
        system_prompt = (
            "You are a flowchart scope assistant. "
            "Return JSON only with keys: assistant_message, scope_summary, "
            "ready_for_diagram (boolean), graph_brief."
        )
        user_prompt = (
            "Current scope summary:\n"
            f"{current_scope or '(none)'}\n\n"
            "Conversation:\n"
            f"{history_text}\n\n"
            "Latest user message:\n"
            f"{user_message}\n\n"
            "Rules:\n"
            "- If requirements are still ambiguous, set ready_for_diagram=false and ask one concrete question.\n"
            "- If requirements are enough, set ready_for_diagram=true and return a concise graph_brief.\n"
            "- Keep assistant_message short and actionable.\n"
        )
        return self.llm_client.complete_json(system_prompt, user_prompt, temperature=0.1)

    def _graph_step(self, graph_brief: str) -> GraphData:
        system_prompt = (
            "You generate flowchart structures. Return JSON only with keys nodes and edges.\n"
            "nodes: [{id,label,type}] where type is input/default/output.\n"
            "edges: [{id,source,target,label}].\n"
            "No coordinates. Keep up to 14 nodes."
        )
        user_prompt = (
            "Create a practical flowchart for the following scope.\n"
            f"Scope:\n{graph_brief}\n"
            "Output strictly valid JSON."
        )
        raw_graph = self.llm_client.complete_json(system_prompt, user_prompt, temperature=0.2)
        return sanitize_graph_data(raw_graph)

    def _update_step(
        self,
        user_message: str,
        chat_history: List[ChatMessage],
        current_scope: str,
        current_graph: GraphData,
    ) -> Dict[str, Any]:
        history_text = _format_history(chat_history[-8:])
        system_prompt = (
            "You update an existing flowchart based on new user requests. "
            "Return JSON only with keys: assistant_message, scope_summary, graph, impact_message."
        )
        user_prompt = (
            "Current scope summary:\n"
            f"{current_scope or '(none)'}\n\n"
            "Current graph JSON:\n"
            f"{json.dumps(current_graph, ensure_ascii=False)}\n\n"
            "Conversation:\n"
            f"{history_text}\n\n"
            "Latest user request:\n"
            f"{user_message}\n\n"
            "Rules:\n"
            "- graph must include full updated nodes/edges (not delta).\n"
            "- Preserve unaffected IDs whenever possible.\n"
            "- Keep graph valid (no dangling edges).\n"
            "- impact_message must explain what changed in one sentence.\n"
        )
        return self.llm_client.complete_json(system_prompt, user_prompt, temperature=0.2)

    def _run_fallback_initial(self, user_message: str, current_scope: str) -> AgentTurn:
        combined_scope = (current_scope + "\n" + user_message).strip()
        if len(combined_scope) < 20:
            return AgentTurn(
                assistant_message=(
                    "Please clarify scope: users, start condition, end condition, and exceptions."
                ),
                scope_summary=current_scope or user_message,
                graph_data=None,
                mode="scope",
                source="fallback",
                impact={
                    "phase": "initial",
                    "message": "Need more scope details before diagram generation.",
                    "impacted_node_ids": [],
                    "impacted_edge_ids": [],
                },
            )

        graph_data = build_mock_graph(user_message)
        return AgentTurn(
            assistant_message="Generated an initial draft diagram from your scope.",
            scope_summary=combined_scope,
            graph_data=graph_data,
            mode="visualize",
            source="fallback",
            impact={
                "phase": "initial",
                "message": "Initial diagram generated.",
                "impacted_node_ids": sorted([node["id"] for node in graph_data["nodes"]]),
                "impacted_edge_ids": sorted([edge["id"] for edge in graph_data["edges"]]),
            },
        )

    def _run_fallback_update(
        self,
        user_message: str,
        current_scope: str,
        current_graph: GraphData,
    ) -> AgentTurn:
        new_scope = (current_scope + "\n" + user_message).strip()
        updated_graph = build_mock_graph(user_message or current_scope or "workflow update")
        impact = compute_impact_range(current_graph, updated_graph)
        impact["phase"] = "update"
        return AgentTurn(
            assistant_message=(
                "Updated the diagram draft. Review impacted nodes to verify side effects."
            ),
            scope_summary=new_scope,
            graph_data=updated_graph,
            mode="visualize",
            source="fallback",
            impact=impact,
        )


class MermaidDiagramOrchestrator:
    def __init__(self, llm_client: Optional[OpenAIJSONClient] = None) -> None:
        self.llm_client = llm_client or OpenAIJSONClient()
        self.observe_agent = MermaidObserveAgent()
        self.act_agent = MermaidActAgent(sanitize_mermaid_code)
        self.verify_agent = MermaidVerifyAgent(sanitize_mermaid_code)
        self.role_coordinator = MermaidRoleCoordinator(sanitize_mermaid_code)

    def run_turn(
        self,
        diagram_type: str,
        user_message: str,
        chat_history: List[ChatMessage],
        current_code: str,
    ) -> MermaidTurn:
        observed = self.observe_agent.observe(user_message=user_message, current_code=current_code)
        has_code = observed.has_current_code

        if not self.llm_client.is_enabled():
            return self._run_fallback_with_roles(
                diagram_type=diagram_type,
                user_message=observed.user_message,
                chat_history=chat_history,
                current_code=current_code,
                reason="LLM disabled.",
            )

        try:
            if has_code:
                turn = self._run_llm_update(diagram_type, user_message, chat_history, current_code)
                ok, reason = self.verify_agent.verify(diagram_type, turn.mermaid_code, current_code, user_message)
                if ok:
                    return turn
                return self._run_fallback_with_roles(
                    diagram_type=diagram_type,
                    user_message=observed.user_message,
                    chat_history=chat_history,
                    current_code=current_code,
                    reason=f"verify failed: {reason}",
                )
            turn = self._run_llm_initial(diagram_type, user_message, chat_history)
            ok, reason = self.verify_agent.verify(diagram_type, turn.mermaid_code, "", user_message)
            if ok:
                return turn
            return self._run_fallback_with_roles(
                diagram_type=diagram_type,
                user_message=observed.user_message,
                chat_history=chat_history,
                current_code=current_code,
                reason=f"verify failed: {reason}",
            )
        except Exception as exc:
            return self._run_fallback_with_roles(
                diagram_type=diagram_type,
                user_message=observed.user_message,
                chat_history=chat_history,
                current_code=current_code,
                reason=f"LLM call error: {exc}",
            )

    def _run_fallback_with_roles(
        self,
        diagram_type: str,
        user_message: str,
        chat_history: List[ChatMessage],
        current_code: str,
        reason: str,
    ) -> MermaidTurn:
        result = self.role_coordinator.run(
            diagram_type=diagram_type,
            user_message=user_message,
            chat_history=chat_history,
            current_code=current_code,
            default_template_code=_default_mermaid_template(diagram_type),
        )
        suffix = f" ({reason})" if reason else ""
        return MermaidTurn(
            assistant_message=f"{result.summary}{suffix}",
            mermaid_code=result.mermaid_code,
            source="fallback",
            phase="update" if bool((current_code or "").strip()) else "initial",
            change_summary=f"{result.summary}{suffix}",
        )

    def _run_llm_initial(
        self,
        diagram_type: str,
        user_message: str,
        chat_history: List[ChatMessage],
    ) -> MermaidTurn:
        history_text = _format_history(chat_history[-8:])
        system_prompt = (
            "You are a Mermaid diagram assistant. "
            "Return JSON only with keys: assistant_message, mermaid_code, change_summary.\n"
            "mermaid_code must be valid Mermaid syntax for the requested diagram type."
        )
        user_prompt = (
            f"Diagram type: {diagram_type}\n\n"
            "Conversation:\n"
            f"{history_text}\n\n"
            "Latest user request:\n"
            f"{user_message}\n\n"
            "Rules:\n"
            "- Output a complete, runnable Mermaid diagram.\n"
            "- Keep it practical and under 24 lines.\n"
            "- Preserve language from user request when possible.\n"
        )
        result = self.llm_client.complete_json(system_prompt, user_prompt, temperature=0.2)
        code = sanitize_mermaid_code(diagram_type, str(result.get("mermaid_code", "")))
        return MermaidTurn(
            assistant_message=str(result.get("assistant_message", "Generated Mermaid draft.")).strip()
            or "Generated Mermaid draft.",
            mermaid_code=code,
            source="llm",
            phase="initial",
            change_summary=str(result.get("change_summary", "Initial draft generated.")).strip()
            or "Initial draft generated.",
        )

    def _run_llm_update(
        self,
        diagram_type: str,
        user_message: str,
        chat_history: List[ChatMessage],
        current_code: str,
    ) -> MermaidTurn:
        history_text = _format_history(chat_history[-8:])
        system_prompt = (
            "You update Mermaid diagram code based on user requests. "
            "Return JSON only with keys: assistant_message, mermaid_code, change_summary."
        )
        user_prompt = (
            f"Diagram type: {diagram_type}\n\n"
            "Current Mermaid code:\n"
            f"{current_code}\n\n"
            "Conversation:\n"
            f"{history_text}\n\n"
            "Latest user request:\n"
            f"{user_message}\n\n"
            "Rules:\n"
            "- Return full updated Mermaid code, not delta.\n"
            "- Keep existing lines unless user asks to change them.\n"
            "- Keep code valid for the diagram type.\n"
            "- change_summary must mention impacted area in one sentence.\n"
        )
        result = self.llm_client.complete_json(system_prompt, user_prompt, temperature=0.2)
        code = sanitize_mermaid_code(diagram_type, str(result.get("mermaid_code", "")), current_code)
        return MermaidTurn(
            assistant_message=str(result.get("assistant_message", "Updated Mermaid diagram.")).strip()
            or "Updated Mermaid diagram.",
            mermaid_code=code,
            source="llm",
            phase="update",
            change_summary=str(result.get("change_summary", "Diagram updated.")).strip()
            or "Diagram updated.",
        )

    def _run_fallback_initial(
        self,
        diagram_type: str,
        user_message: str,
        reason: str = "",
    ) -> MermaidTurn:
        fallback_code = self.act_agent.fallback_code(
            diagram_type=diagram_type,
            user_message=user_message,
            current_code="",
            append=False,
            default_template_code=_default_mermaid_template(diagram_type),
        )
        if not fallback_code.strip():
            fallback_code = _default_mermaid_template(diagram_type)
        reason_suffix = f" ({reason})" if reason else ""
        return MermaidTurn(
            assistant_message=f"Generated draft in fallback mode using request text{reason_suffix}",
            mermaid_code=fallback_code,
            source="fallback",
            phase="initial",
            change_summary=f"Fallback initial draft generated{reason_suffix}.",
        )

    def _run_fallback_update(
        self,
        diagram_type: str,
        user_message: str,
        current_code: str,
        reason: str = "",
    ) -> MermaidTurn:
        next_code = self.act_agent.fallback_code(
            diagram_type=diagram_type,
            user_message=user_message,
            current_code=current_code,
            append=True,
            default_template_code=_default_mermaid_template(diagram_type),
        )
        reason_suffix = f" ({reason})" if reason else ""
        return MermaidTurn(
            assistant_message=f"Switched to fallback update after verify/check{reason_suffix}.",
            mermaid_code=next_code,
            source="fallback",
            phase="update",
            change_summary=f"Fallback update applied with observe-act-verify cycle{reason_suffix}.",
        )


def sanitize_graph_data(raw: Dict[str, Any]) -> GraphData:
    raw_nodes = raw.get("nodes", []) if isinstance(raw, dict) else []
    raw_edges = raw.get("edges", []) if isinstance(raw, dict) else []

    nodes: List[Dict[str, str]] = []
    node_ids = set()
    for index, node in enumerate(raw_nodes, start=1):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip() or f"node_{index}"
        if node_id in node_ids:
            node_id = f"{node_id}_{index}"
        node_ids.add(node_id)
        node_type = str(node.get("type", "default")).strip() or "default"
        if node_type not in {"input", "default", "output"}:
            node_type = "default"
        label = str(node.get("label", node_id)).strip() or node_id
        nodes.append({"id": node_id, "label": label, "type": node_type})

    edges: List[Dict[str, str]] = []
    for index, edge in enumerate(raw_edges, start=1):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source not in node_ids or target not in node_ids:
            continue
        edge_id = str(edge.get("id", "")).strip() or f"e{index}"
        label = str(edge.get("label", "")).strip()
        edges.append({"id": edge_id, "source": source, "target": target, "label": label})

    if not nodes:
        return build_mock_graph("default workflow")
    if not edges and len(nodes) >= 2:
        for i in range(len(nodes) - 1):
            edges.append(
                {
                    "id": f"auto_e{i+1}",
                    "source": nodes[i]["id"],
                    "target": nodes[i + 1]["id"],
                    "label": "",
                }
            )

    return {"nodes": nodes, "edges": edges}


def compute_impact_range(previous_graph: GraphData, updated_graph: GraphData) -> Dict[str, Any]:
    prev_nodes = {node["id"]: node for node in previous_graph.get("nodes", [])}
    next_nodes = {node["id"]: node for node in updated_graph.get("nodes", [])}
    prev_edges = {edge["id"]: edge for edge in previous_graph.get("edges", [])}
    next_edges = {edge["id"]: edge for edge in updated_graph.get("edges", [])}

    prev_node_ids = set(prev_nodes.keys())
    next_node_ids = set(next_nodes.keys())
    prev_edge_ids = set(prev_edges.keys())
    next_edge_ids = set(next_edges.keys())

    added_node_ids = sorted(next_node_ids - prev_node_ids)
    removed_node_ids = sorted(prev_node_ids - next_node_ids)
    changed_node_ids = sorted(
        node_id
        for node_id in (prev_node_ids & next_node_ids)
        if (
            prev_nodes[node_id].get("label") != next_nodes[node_id].get("label")
            or prev_nodes[node_id].get("type") != next_nodes[node_id].get("type")
        )
    )

    added_edge_ids = sorted(next_edge_ids - prev_edge_ids)
    removed_edge_ids = sorted(prev_edge_ids - next_edge_ids)
    changed_edge_ids = sorted(
        edge_id
        for edge_id in (prev_edge_ids & next_edge_ids)
        if (
            prev_edges[edge_id].get("source") != next_edges[edge_id].get("source")
            or prev_edges[edge_id].get("target") != next_edges[edge_id].get("target")
            or prev_edges[edge_id].get("label", "") != next_edges[edge_id].get("label", "")
        )
    )

    impacted_node_ids = set(added_node_ids + removed_node_ids + changed_node_ids)
    for edge_id in added_edge_ids + changed_edge_ids:
        edge = next_edges.get(edge_id)
        if edge:
            impacted_node_ids.add(edge.get("source", ""))
            impacted_node_ids.add(edge.get("target", ""))
    for edge_id in removed_edge_ids:
        edge = prev_edges.get(edge_id)
        if edge:
            impacted_node_ids.add(edge.get("source", ""))
            impacted_node_ids.add(edge.get("target", ""))

    impacted_node_ids.discard("")

    impacted_edge_ids = sorted(set(added_edge_ids + removed_edge_ids + changed_edge_ids))

    return {
        "phase": "update",
        "message": (
            f"nodes(+{len(added_node_ids)}, -{len(removed_node_ids)}, ~{len(changed_node_ids)}), "
            f"edges(+{len(added_edge_ids)}, -{len(removed_edge_ids)}, ~{len(changed_edge_ids)})"
        ),
        "added_node_ids": added_node_ids,
        "removed_node_ids": removed_node_ids,
        "changed_node_ids": changed_node_ids,
        "added_edge_ids": added_edge_ids,
        "removed_edge_ids": removed_edge_ids,
        "changed_edge_ids": changed_edge_ids,
        "impacted_node_ids": sorted(impacted_node_ids),
        "impacted_edge_ids": impacted_edge_ids,
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not include JSON object.")
    return json.loads(text[start : end + 1])


def _format_history(history: List[ChatMessage]) -> str:
    lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no history)"


MERMAID_HEADERS = {
    "Flowchart": "graph TD",
    "Sequence": "sequenceDiagram",
    "State": "stateDiagram-v2",
    "ER": "erDiagram",
    "Class": "classDiagram",
    "Gantt": "gantt",
}


def sanitize_mermaid_code(diagram_type: str, raw_code: str, fallback_code: str = "") -> str:
    code = _strip_code_fence(raw_code or "")
    if not code.strip():
        code = _strip_code_fence(fallback_code or "")

    expected_header = MERMAID_HEADERS.get(diagram_type, "").strip()
    stripped = code.strip()
    if not stripped:
        stripped = expected_header

    first_line = stripped.splitlines()[0].strip() if stripped.splitlines() else ""
    if expected_header and first_line != expected_header:
        stripped = f"{expected_header}\n{stripped}"

    return stripped.rstrip() + "\n"


def _strip_code_fence(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _default_mermaid_template(diagram_type: str) -> str:
    templates = list_mermaid_templates(diagram_type)
    if templates:
        return sanitize_mermaid_code(diagram_type, get_mermaid_template(diagram_type, templates[0]["id"]))
    return sanitize_mermaid_code(diagram_type, MERMAID_HEADERS.get(diagram_type, "graph TD"))
