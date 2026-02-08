import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .canvas_graph import graph_to_mermaid, parse_mermaid_to_graph


@dataclass
class ObserveResult:
    has_current_code: bool
    user_message: str
    steps: List[str]


class MermaidObserveAgent:
    def observe(self, user_message: str, current_code: str) -> ObserveResult:
        return ObserveResult(
            has_current_code=bool((current_code or "").strip()),
            user_message=(user_message or "").strip(),
            steps=self.extract_steps(user_message),
        )

    def extract_steps(self, text: str) -> List[str]:
        raw = (text or "").strip()
        if not raw:
            return []

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        steps: List[str] = []
        for line in lines:
            cleaned = re.sub(r"^[-*]\s*", "", line)
            cleaned = re.sub(r"^\d+[\).]\s*", "", cleaned)
            cleaned = re.sub(r"^#+\s*", "", cleaned).strip()
            if cleaned:
                steps.extend([item.strip() for item in re.split(r"[。.!?]", cleaned) if item.strip()])

        if not steps:
            steps = [raw]
        return steps[:8]


class MermaidActAgent:
    def __init__(self, sanitize_mermaid_code: Callable[[str, str, str], str]) -> None:
        self._sanitize_mermaid_code = sanitize_mermaid_code
        self._observe = MermaidObserveAgent()

    def fallback_code(
        self,
        diagram_type: str,
        user_message: str,
        current_code: str,
        append: bool,
        default_template_code: str,
    ) -> str:
        seed = (
            parse_mermaid_to_graph(diagram_type, current_code) if (current_code or "").strip() else {"nodes": [], "edges": []}
        )
        steps = self._observe.extract_steps(user_message)
        if not steps:
            if append and (current_code or "").strip():
                return self._sanitize_mermaid_code(diagram_type, current_code, "")
            return self._sanitize_mermaid_code(diagram_type, default_template_code, "")

        graph = {"nodes": list(seed.get("nodes", [])), "edges": list(seed.get("edges", []))}
        known_ids = {node.get("id", "") for node in graph["nodes"]}

        if len(graph["nodes"]) == 0:
            graph["nodes"].append({"id": "start", "label": steps[0], "type": "input"})
            known_ids.add("start")
            previous_id = "start"
            step_slice = steps[1:]
        else:
            previous_id = str(graph["nodes"][-1].get("id", "start")) or "start"
            step_slice = steps

        for index, step in enumerate(step_slice, start=1):
            node_id = f"step_{index}"
            while node_id in known_ids:
                node_id = f"{node_id}_n"
            known_ids.add(node_id)
            graph["nodes"].append({"id": node_id, "label": step, "type": "default"})
            graph["edges"].append(
                {
                    "id": f"e_fallback_{len(graph['edges']) + 1}",
                    "source": previous_id,
                    "target": node_id,
                    "label": step if diagram_type == "Sequence" else "",
                }
            )
            previous_id = node_id

        return self._sanitize_mermaid_code(diagram_type, graph_to_mermaid(diagram_type, graph), current_code)


class MermaidVerifyAgent:
    def __init__(self, sanitize_mermaid_code: Callable[[str, str, str], str]) -> None:
        self._sanitize_mermaid_code = sanitize_mermaid_code

    def verify(
        self,
        diagram_type: str,
        candidate_code: str,
        current_code: str,
        user_message: str,
    ) -> Tuple[bool, str]:
        candidate = self._sanitize_mermaid_code(diagram_type, candidate_code, current_code)
        if not candidate.strip():
            return False, "empty candidate"

        if user_message.strip() and current_code.strip():
            prev = self._sanitize_mermaid_code(diagram_type, current_code, "")
            if self._normalize(prev) == self._normalize(candidate):
                return False, "no effective update"

        parsed = parse_mermaid_to_graph(diagram_type, candidate)
        if len(parsed.get("nodes", [])) == 0:
            return False, "no nodes parsed"
        return True, "ok"

    def _normalize(self, code: str) -> str:
        return "\n".join(line.strip() for line in (code or "").splitlines() if line.strip())
