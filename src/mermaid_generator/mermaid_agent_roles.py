import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .canvas_graph import graph_to_mermaid, parse_mermaid_to_graph

MERMAID_HEADERS = {
    "Flowchart": "graph TD",
    "Sequence": "sequenceDiagram",
    "State": "stateDiagram-v2",
    "ER": "erDiagram",
    "Class": "classDiagram",
    "Gantt": "gantt",
}


@dataclass
class ObserveResult:
    has_current_code: bool
    user_message: str
    steps: List[str]

    def validate(self) -> Tuple[bool, str]:
        if self.user_message == "":
            return False, "empty user_message"
        return True, "ok"


@dataclass
class MermaidOutputModel:
    diagram_type: str
    mermaid_code: str
    node_count: int
    edge_count: int

    @classmethod
    def from_code(
        cls,
        diagram_type: str,
        candidate_code: str,
        fallback_code: str,
        sanitize_mermaid_code: Callable[[str, str, str], str],
    ) -> Tuple[Optional["MermaidOutputModel"], str]:
        candidate = sanitize_mermaid_code(diagram_type, candidate_code, fallback_code)
        if not candidate.strip():
            return None, "empty candidate"

        header = MERMAID_HEADERS.get(diagram_type, "").strip()
        lines = [line.rstrip() for line in candidate.splitlines()]
        first_line = lines[0].strip() if lines else ""
        if header and first_line != header:
            return None, "header mismatch"

        body_lines = [line.strip() for line in lines[1:] if line.strip() and not line.strip().startswith("%%")]
        if len(body_lines) == 0:
            return None, "missing body"
        if not _has_type_specific_mermaid_content(diagram_type, body_lines):
            return None, "missing type-specific syntax"

        parsed = parse_mermaid_to_graph(diagram_type, candidate)
        node_count = len(parsed.get("nodes", []))
        edge_count = len(parsed.get("edges", []))
        if node_count == 0:
            return None, "no nodes parsed"

        return (
            cls(
                diagram_type=diagram_type,
                mermaid_code=candidate,
                node_count=node_count,
                edge_count=edge_count,
            ),
            "ok",
        )


def _has_type_specific_mermaid_content(diagram_type: str, body_lines: List[str]) -> bool:
    body = "\n".join(body_lines)
    if diagram_type == "Flowchart":
        return "-->" in body or "[" in body
    if diagram_type == "Sequence":
        return "participant " in body or "->>" in body or "-->>" in body or "-)" in body
    if diagram_type == "State":
        return "-->" in body
    if diagram_type == "ER":
        return "--" in body
    if diagram_type == "Class":
        return "class " in body or "-->" in body or "<|--" in body or "..>" in body
    if diagram_type == "Gantt":
        return "section " in body or ":" in body
    return True


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
        model, reason = self.verify_output(
            diagram_type=diagram_type,
            candidate_code=candidate_code,
            fallback_code=current_code,
        )
        if not model:
            return False, reason

        if user_message.strip() and current_code.strip():
            prev = self._sanitize_mermaid_code(diagram_type, current_code, "")
            if normalize_mermaid(prev) == normalize_mermaid(model.mermaid_code):
                return False, "no effective update"
        return True, "ok"

    def verify_output(
        self,
        diagram_type: str,
        candidate_code: str,
        fallback_code: str = "",
    ) -> Tuple[Optional[MermaidOutputModel], str]:
        return MermaidOutputModel.from_code(
            diagram_type=diagram_type,
            candidate_code=candidate_code,
            fallback_code=fallback_code,
            sanitize_mermaid_code=self._sanitize_mermaid_code,
        )


@dataclass
class InterpreterObservation:
    user_message: str
    chat_history: List[Dict[str, str]]
    memory_lines: List[str]

    def validate(self) -> Tuple[bool, str]:
        if self.user_message == "":
            return False, "empty user_message"
        return True, "ok"


@dataclass
class InterpreterArtifact:
    normalized_request: str
    goals: List[str]
    constraints: List[str]
    ambiguity_flags: List[str]
    memory_lines: List[str]

    def validate(self) -> Tuple[bool, str]:
        if not self.normalized_request.strip():
            return False, "empty normalized_request"
        if len(self.goals) == 0 and len(self.ambiguity_flags) == 0:
            return False, "missing goal analysis"
        return True, "ok"


class InterpreterRoleAgent:
    def observe(self, user_message: str, chat_history: List[Dict[str, str]]) -> InterpreterObservation:
        normalized = (user_message or "").strip()
        memory_lines = parse_memory_lines(normalized)
        observation = InterpreterObservation(
            user_message=normalized,
            chat_history=list(chat_history or []),
            memory_lines=memory_lines,
        )
        return observation

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
        return artifact.validate()


@dataclass
class ArchitectObservation:
    diagram_type: str
    current_code: str
    interpreter: InterpreterArtifact

    def validate(self) -> Tuple[bool, str]:
        if not self.diagram_type.strip():
            return False, "missing diagram_type"
        ok, reason = self.interpreter.validate()
        if not ok:
            return False, f"interpreter-invalid:{reason}"
        return True, "ok"


@dataclass
class ArchitectArtifact:
    plan_steps: List[str]
    invariants: List[str]
    impacted_regions: List[str]
    strategy: str

    def validate(self) -> Tuple[bool, str]:
        if len(self.plan_steps) == 0:
            return False, "empty plan"
        if len(self.invariants) == 0:
            return False, "missing invariants"
        if self.strategy not in {"create", "append"}:
            return False, "invalid strategy"
        return True, "ok"


class ArchitectRoleAgent:
    def observe(
        self,
        diagram_type: str,
        current_code: str,
        interpreter: InterpreterArtifact,
    ) -> ArchitectObservation:
        observation = ArchitectObservation(
            diagram_type=diagram_type,
            current_code=current_code or "",
            interpreter=interpreter,
        )
        return observation

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
        return artifact.validate()


@dataclass
class SystemEngineerObservation:
    diagram_type: str
    current_code: str
    architecture: ArchitectArtifact

    def validate(self) -> Tuple[bool, str]:
        if not self.diagram_type.strip():
            return False, "missing diagram_type"
        ok, reason = self.architecture.validate()
        if not ok:
            return False, f"architecture-invalid:{reason}"
        return True, "ok"


@dataclass
class SystemEngineerArtifact:
    mermaid_code: str
    validation_hints: List[str]
    applied_steps: int

    def validate(self) -> Tuple[bool, str]:
        if self.applied_steps <= 0:
            return False, "no applied steps"
        if not self.mermaid_code.strip():
            return False, "empty mermaid code"
        return True, "ok"


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
        observation = SystemEngineerObservation(
            diagram_type=diagram_type,
            current_code=current_code or "",
            architecture=architecture,
        )
        return observation

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
        ok, reason = artifact.validate()
        if not ok:
            return False, reason
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
        self._final_verify_agent = MermaidVerifyAgent(sanitize_mermaid_code)

    def run(
        self,
        diagram_type: str,
        user_message: str,
        chat_history: List[Dict[str, str]],
        current_code: str,
        default_template_code: str,
    ) -> RoleCoordinationResult:
        interpreter_obs = self.interpreter.observe(user_message=user_message, chat_history=chat_history)
        ok, reason = interpreter_obs.validate()
        if not ok:
            return self._fallback(
                diagram_type=diagram_type,
                user_message=user_message,
                current_code=current_code,
                default_template_code=default_template_code,
                failed_role=f"interpreter_observation:{reason}",
            )
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
        ok, reason = architect_obs.validate()
        if not ok:
            return self._fallback(
                diagram_type=diagram_type,
                user_message=user_message,
                current_code=current_code,
                default_template_code=default_template_code,
                failed_role=f"architect_observation:{reason}",
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
        ok, reason = engineer_obs.validate()
        if not ok:
            return self._fallback(
                diagram_type=diagram_type,
                user_message=user_message,
                current_code=current_code,
                default_template_code=default_template_code,
                failed_role=f"system_engineer_observation:{reason}",
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

        output_model, reason = self._final_verify_agent.verify_output(
            diagram_type=diagram_type,
            candidate_code=engineer_artifact.mermaid_code,
            fallback_code=default_template_code,
        )
        if not output_model:
            return self._fallback(
                diagram_type=diagram_type,
                user_message=user_message,
                current_code=current_code,
                default_template_code=default_template_code,
                failed_role=f"final_output:{reason}",
            )

        return RoleCoordinationResult(
            mermaid_code=output_model.mermaid_code,
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
        generated = self._fallback_actor.fallback_code(
            diagram_type=diagram_type,
            user_message=user_message,
            current_code=current_code,
            append=append,
            default_template_code=default_template_code,
        )
        output_model, reason = self._final_verify_agent.verify_output(
            diagram_type=diagram_type,
            candidate_code=generated,
            fallback_code=default_template_code,
        )
        if not output_model:
            fallback_model, _ = self._final_verify_agent.verify_output(
                diagram_type=diagram_type,
                candidate_code=default_template_code,
                fallback_code=default_template_code,
            )
            output_model = fallback_model
            failed_role = f"{failed_role};fallback_output:{reason}"

        return RoleCoordinationResult(
            mermaid_code=output_model.mermaid_code if output_model else default_template_code,
            summary=f"Role pipeline fallback applied due to {failed_role}.",
            source="fallback",
            failed_role=failed_role,
        )


def extract_steps(text: str) -> List[str]:
    raw, _ = split_user_and_memory(text)
    if not raw.strip():
        return []

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    steps: List[str] = []
    for line in lines:
        cleaned = re.sub(r"^[-*]\s*", "", line)
        cleaned = re.sub(r"^\d+[\).]\s*", "", cleaned)
        cleaned = re.sub(r"^#+\s*", "", cleaned).strip()
        cleaned = re.sub(r"^>\s*", "", cleaned).strip()
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
        if not cleaned:
            continue
        if cleaned.startswith("|") and cleaned.endswith("|"):
            continue
        if re.fullmatch(r"[-:| ]+", cleaned):
            continue
        for item in [part.strip() for part in re.split(r"[。.!?]", cleaned) if part.strip()]:
            normalized = re.sub(r"\s+", " ", item).strip()
            if len(normalized) > 80:
                normalized = f"{normalized[:77].rstrip()}..."
            if normalized:
                steps.append(normalized)

    if not steps:
        fallback = re.sub(r"\s+", " ", raw).strip()
        if len(fallback) > 80:
            fallback = f"{fallback[:77].rstrip()}..."
        steps = [fallback] if fallback else []
    return steps[:8]


def parse_memory_lines(user_message: str) -> List[str]:
    _, memory = split_user_and_memory(user_message)
    if not memory:
        return []
    lines = [line.strip() for line in memory.splitlines() if line.strip()]
    return [line for line in lines if line.startswith("- ")]


def split_user_and_memory(user_message: str) -> Tuple[str, str]:
    raw = (user_message or "")
    marker = "Session Memory:"
    if marker not in raw:
        return raw.strip(), ""
    head, tail = raw.split(marker, 1)
    return head.strip(), tail.strip()


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
