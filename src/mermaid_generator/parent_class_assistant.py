from typing import Any, Dict, List, Optional

from .orchestrator import OpenAIJSONClient
from .property_editor import parse_node_properties

GraphData = Dict[str, List[Dict[str, str]]]


class ParentClassAssistant:
    def __init__(self, llm_client: Optional[OpenAIJSONClient] = None) -> None:
        self.llm_client = llm_client or OpenAIJSONClient()

    def suggest_parent(
        self,
        graph_data: GraphData,
        child_id: str,
        user_request: str = "",
    ) -> Dict[str, str]:
        child_id = str(child_id or "").strip()
        if not child_id:
            raise ValueError("child_id is required")

        class_nodes = self._class_nodes(graph_data)
        candidates = [node for node in class_nodes if node["id"] != child_id]
        if not candidates:
            return {
                "child_id": child_id,
                "parent_id": "",
                "reason": "No parent candidate is available.",
                "source": "fallback",
            }

        if self.llm_client.is_enabled():
            try:
                suggestion = self._suggest_with_llm(class_nodes, child_id, user_request)
                parent_id = suggestion.get("parent_id", "")
                if parent_id in {node["id"] for node in candidates}:
                    return {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "reason": suggestion.get("reason", "LLM suggested parent class."),
                        "source": "llm",
                    }
            except Exception as exc:
                fallback = self._fallback(candidates, child_id)
                fallback["reason"] += f" (LLM error: {exc})"
                return fallback

        return self._fallback(candidates, child_id)

    def _class_nodes(self, graph_data: GraphData) -> List[Dict[str, str]]:
        nodes = []
        for node in graph_data.get("nodes", []):
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            node_name = parse_node_properties("Class", node_id, str(node.get("label", ""))).get("name", node_id)
            nodes.append({"id": node_id, "name": node_name})
        return nodes

    def _suggest_with_llm(
        self, class_nodes: List[Dict[str, str]], child_id: str, user_request: str
    ) -> Dict[str, str]:
        lines = [f"- {node['id']}: {node['name']}" for node in class_nodes]
        system_prompt = (
            "You are a class design assistant. Return JSON only with keys: parent_id, reason.\n"
            "parent_id must be one of provided class ids and must not equal child_id."
        )
        user_prompt = (
            f"child_id: {child_id}\n"
            f"user_request: {user_request or '(none)'}\n"
            "class_nodes:\n"
            f"{chr(10).join(lines)}\n"
            "Pick the best parent class for inheritance."
        )
        payload = self.llm_client.complete_json(system_prompt, user_prompt, temperature=0.1)
        return {
            "parent_id": str(payload.get("parent_id", "")).strip(),
            "reason": str(payload.get("reason", "")).strip() or "LLM suggested parent class.",
        }

    def _fallback(self, candidates: List[Dict[str, str]], child_id: str) -> Dict[str, str]:
        selected = sorted(candidates, key=lambda item: (item["name"].lower(), item["id"]))[0]
        return {
            "child_id": child_id,
            "parent_id": selected["id"],
            "reason": f"Fallback selected '{selected['id']}' as parent based on deterministic ordering.",
            "source": "fallback",
        }
