import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
            steps=extract_steps(user_message),
        )


class MermaidActAgent:
    def __init__(self, sanitize_mermaid_code: Callable[[str, str, str], str]) -> None:
        self._sanitize_mermaid_code = sanitize_mermaid_code

    def fallback_code(
        self,
        diagram_type: str,
        user_message: str,
        current_code: str,
        append: bool,
        default_template_code: str,
    ) -> str:
        return build_mermaid_from_steps(
            diagram_type=diagram_type,
            user_message=user_message,
            current_code=current_code,
            append=append,
            default_template_code=default_template_code,
            sanitize_mermaid_code=self._sanitize_mermaid_code,
        )


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
            if normalize_mermaid(prev) == normalize_mermaid(candidate):
                return False, "no effective update"

        parsed = parse_mermaid_to_graph(diagram_type, candidate)
        if len(parsed.get("nodes", [])) == 0:
            return False, "no nodes parsed"
        return True, "ok"


@dataclass
class InterpreterObservation:
    user_message: str
    chat_history: List[Dict[str, str]]
    memory_lines: List[str]


@dataclass
class InterpreterArtifact:
    normalized_request: str
    goals: List[str]
    constraints: List[str]
    ambiguity_flags: List[str]
    memory_lines: List[str]


class InterpreterRoleAgent:
    def observe(self, user_message: str, chat_history: List[Dict[str, str]]) -> InterpreterObservation:
        normalized = (user_message or "").strip()
        memory_lines = parse_memory_lines(normalized)
        return InterpreterObservation(
            user_message=normalized,
            chat_history=list(chat_history or []),
            memory_lines=memory_lines,
        )

    def react(self, observation: InterpreterObservation) -> InterpreterArtifact:
        goals = extract_steps(observation.user_message)
        constraints = []
        if "must" in observation.user_message.lower():
            constraints.append("must-clause-detected")
        if "avoid" in observation.user_message.lower():
            constraints.append("avoid-clause-detected")
        ambiguity_flags = []
        if len(goals) == 0:
            ambiguity_flags.append("no-goal-detected")
        return InterpreterArtifact(
            normalized_request=observation.user_message,
            goals=goals,
            constraints=constraints,
            ambiguity_flags=ambiguity_flags,
            memory_lines=observation.memory_lines,
        )

    def verify(self, artifact: InterpreterArtifact) -> Tuple[bool, str]:
        if not artifact.normalized_request.strip():
            return False, "empty request"
        if len(artifact.goals) == 0 and len(artifact.ambiguity_flags) == 0:
            return False, "missing goal analysis"
        return True, "ok"


@dataclass
class ArchitectObservation:
    diagram_type: str
    current_code: str
    interpreter: InterpreterArtifact


@dataclass
class ArchitectArtifact:
    plan_steps: List[str]
    invariants: List[str]
    impacted_regions: List[str]
    strategy: str


class ArchitectRoleAgent:
    def observe(
        self,
        diagram_type: str,
        current_code: str,
        interpreter: InterpreterArtifact,
    ) -> ArchitectObservation:
        return ArchitectObservation(
            diagram_type=diagram_type,
            current_code=current_code or "",
            interpreter=interpreter,
        )

    def react(self, observation: ArchitectObservation) -> ArchitectArtifact:
        parsed = parse_mermaid_to_graph(observation.diagram_type, observation.current_code or "")
        plan_steps = list(observation.interpreter.goals[:])
        if len(plan_steps) == 0:
            plan_steps = ["Clarify scope", "Propose draft flow"]

        invariants = ["valid-mermaid", "preserve-unrelated-structure-when-possible"]
        if observation.interpreter.memory_lines:
            invariants.append("respect-template-memory")

        impacted_regions = [node.get("id", "") for node in parsed.get("nodes", [])[:5] if node.get("id")]
        if not impacted_regions:
            impacted_regions = ["new-diagram"]

        strategy = "append" if bool((observation.current_code or "").strip()) else "create"
        return ArchitectArtifact(
            plan_steps=plan_steps,
            invariants=invariants,
            impacted_regions=impacted_regions,
            strategy=strategy,
        )

    def verify(self, artifact: ArchitectArtifact) -> Tuple[bool, str]:
        if len(artifact.plan_steps) == 0:
            return False, "empty plan"
        if len(artifact.invariants) == 0:
            return False, "missing invariants"
        return True, "ok"


@dataclass
class SystemEngineerObservation:
    diagram_type: str
    current_code: str
    architecture: ArchitectArtifact


@dataclass
class SystemEngineerArtifact:
    mermaid_code: str
    validation_hints: List[str]
    applied_steps: int


class SystemEngineerRoleAgent:
    def __init__(self, sanitize_mermaid_code: Callable[[str, str, str], str]) -> None:
        self._sanitize_mermaid_code = sanitize_mermaid_code
        self._verify_agent = MermaidVerifyAgent(sanitize_mermaid_code)

    def observe(
        self,
        diagram_type: str,
        current_code: str,
        architecture: ArchitectArtifact,
    ) -> SystemEngineerObservation:
        return SystemEngineerObservation(
            diagram_type=diagram_type,
            current_code=current_code or "",
            architecture=architecture,
        )

    def react(self, observation: SystemEngineerObservation, default_template_code: str) -> SystemEngineerArtifact:
        request = ". ".join(observation.architecture.plan_steps)
        append = observation.architecture.strategy == "append"
        code = build_mermaid_from_steps(
            diagram_type=observation.diagram_type,
            user_message=request,
            current_code=observation.current_code,
            append=append,
            default_template_code=default_template_code,
            sanitize_mermaid_code=self._sanitize_mermaid_code,
        )
        hints = [f"invariant:{item}" for item in observation.architecture.invariants]
        return SystemEngineerArtifact(
            mermaid_code=code,
            validation_hints=hints,
            applied_steps=len(observation.architecture.plan_steps),
        )

    def verify(
        self,
        diagram_type: str,
        artifact: SystemEngineerArtifact,
        current_code: str,
        user_message: str,
    ) -> Tuple[bool, str]:
        if artifact.applied_steps <= 0:
            return False, "no applied steps"
        return self._verify_agent.verify(
            diagram_type=diagram_type,
            candidate_code=artifact.mermaid_code,
            current_code=current_code,
            user_message=user_message,
        )


@dataclass
class RoleCoordinationResult:
    mermaid_code: str
    summary: str
    source: str
    failed_role: str


class MermaidRoleCoordinator:
    def __init__(
        self,
        sanitize_mermaid_code: Callable[[str, str, str], str],
        interpreter: Optional[InterpreterRoleAgent] = None,
        architect: Optional[ArchitectRoleAgent] = None,
        system_engineer: Optional[SystemEngineerRoleAgent] = None,
    ) -> None:
        self._sanitize_mermaid_code = sanitize_mermaid_code
        self.interpreter = interpreter or InterpreterRoleAgent()
        self.architect = architect or ArchitectRoleAgent()
        self.system_engineer = system_engineer or SystemEngineerRoleAgent(sanitize_mermaid_code)
        self._fallback_actor = MermaidActAgent(sanitize_mermaid_code)

    def run(
        self,
        diagram_type: str,
        user_message: str,
        chat_history: List[Dict[str, str]],
        current_code: str,
        default_template_code: str,
    ) -> RoleCoordinationResult:
        interpreter_obs = self.interpreter.observe(user_message=user_message, chat_history=chat_history)
        interpreter_artifact = self.interpreter.react(interpreter_obs)
        ok, reason = self.interpreter.verify(interpreter_artifact)
        if not ok:
            return self._fallback(
                diagram_type=diagram_type,
                user_message=user_message,
                current_code=current_code,
                default_template_code=default_template_code,
                failed_role=f"interpreter:{reason}",
            )

        architect_obs = self.architect.observe(
            diagram_type=diagram_type,
            current_code=current_code,
            interpreter=interpreter_artifact,
        )
        architect_artifact = self.architect.react(architect_obs)
        ok, reason = self.architect.verify(architect_artifact)
        if not ok:
            return self._fallback(
                diagram_type=diagram_type,
                user_message=user_message,
                current_code=current_code,
                default_template_code=default_template_code,
                failed_role=f"architect:{reason}",
            )

        engineer_obs = self.system_engineer.observe(
            diagram_type=diagram_type,
            current_code=current_code,
            architecture=architect_artifact,
        )
        engineer_artifact = self.system_engineer.react(
            observation=engineer_obs,
            default_template_code=default_template_code,
        )
        ok, reason = self.system_engineer.verify(
            diagram_type=diagram_type,
            artifact=engineer_artifact,
            current_code=current_code,
            user_message=user_message,
        )
        if not ok:
            return self._fallback(
                diagram_type=diagram_type,
                user_message=user_message,
                current_code=current_code,
                default_template_code=default_template_code,
                failed_role=f"system_engineer:{reason}",
            )

        return RoleCoordinationResult(
            mermaid_code=engineer_artifact.mermaid_code,
            summary="3-role pipeline completed: interpreter -> architect -> system engineer.",
            source="roles",
            failed_role="",
        )

    def _fallback(
        self,
        diagram_type: str,
        user_message: str,
        current_code: str,
        default_template_code: str,
        failed_role: str,
    ) -> RoleCoordinationResult:
        append = bool((current_code or "").strip())
        code = self._fallback_actor.fallback_code(
            diagram_type=diagram_type,
            user_message=user_message,
            current_code=current_code,
            append=append,
            default_template_code=default_template_code,
        )
        return RoleCoordinationResult(
            mermaid_code=code,
            summary=f"Role pipeline fallback applied due to {failed_role}.",
            source="fallback",
            failed_role=failed_role,
        )


def extract_steps(text: str) -> List[str]:
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


def parse_memory_lines(user_message: str) -> List[str]:
    raw = (user_message or "")
    marker = "Session Memory:"
    if marker not in raw:
        return []
    tail = raw.split(marker, 1)[1]
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return [line for line in lines if line.startswith("- ")]


def normalize_mermaid(code: str) -> str:
    return "\n".join(line.strip() for line in (code or "").splitlines() if line.strip())


def build_mermaid_from_steps(
    diagram_type: str,
    user_message: str,
    current_code: str,
    append: bool,
    default_template_code: str,
    sanitize_mermaid_code: Callable[[str, str, str], str],
) -> str:
    seed = parse_mermaid_to_graph(diagram_type, current_code) if (current_code or "").strip() else {"nodes": [], "edges": []}
    steps = extract_steps(user_message)
    if not steps:
        if append and (current_code or "").strip():
            return sanitize_mermaid_code(diagram_type, current_code, "")
        return sanitize_mermaid_code(diagram_type, default_template_code, "")

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

    return sanitize_mermaid_code(diagram_type, graph_to_mermaid(diagram_type, graph), current_code)
