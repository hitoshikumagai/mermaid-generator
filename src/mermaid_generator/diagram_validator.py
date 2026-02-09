import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol


MERMAID_HEADERS = {
    "Flowchart": "graph TD",
    "Sequence": "sequenceDiagram",
    "State": "stateDiagram-v2",
    "ER": "erDiagram",
    "Class": "classDiagram",
    "Gantt": "gantt",
}


HEADER_PREFIXES = (
    "graph ",
    "sequenceDiagram",
    "stateDiagram-v2",
    "erDiagram",
    "classDiagram",
    "gantt",
)


class JSONCompletionClient(Protocol):
    def is_enabled(self) -> bool:  # pragma: no cover - runtime integration path
        ...

    def complete_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict:
        ...


@dataclass(frozen=True)
class ValidationFinding:
    severity: str
    rule_id: str
    message: str
    target: str = ""


@dataclass
class ValidationReport:
    source: str
    findings: List[ValidationFinding] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationFinding]:
        return [item for item in self.findings if item.severity == "error"]

    @property
    def warnings(self) -> List[ValidationFinding]:
        return [item for item in self.findings if item.severity == "warning"]

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def short_reason(self) -> str:
        if self.valid and not self.warnings:
            return "ok"
        if not self.valid:
            return "; ".join(item.message for item in self.errors[:3])
        return "; ".join(item.message for item in self.warnings[:2])


class MermaidDiagramValidator:
    def __init__(
        self,
        llm_client: Optional[JSONCompletionClient] = None,
        enable_llm_assist: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.enable_llm_assist = enable_llm_assist

    def validate_turn(
        self,
        diagram_type: str,
        candidate_code: str,
        current_code: str,
        user_message: str,
    ) -> ValidationReport:
        findings = self._deterministic_checks(diagram_type, candidate_code)
        source = "deterministic"
        llm_findings = self._llm_assisted_checks(
            diagram_type=diagram_type,
            candidate_code=candidate_code,
            current_code=current_code,
            user_message=user_message,
        )
        if llm_findings:
            findings.extend(llm_findings)
            source = "llm+deterministic"
        return ValidationReport(source=source, findings=findings)

    def _llm_assisted_checks(
        self,
        diagram_type: str,
        candidate_code: str,
        current_code: str,
        user_message: str,
    ) -> List[ValidationFinding]:
        if not self.enable_llm_assist or self.llm_client is None:
            return []
        try:
            if not self.llm_client.is_enabled():
                return []
        except Exception:
            return []

        system_prompt = (
            "You are a Mermaid validator. Return JSON only: "
            '{"findings":[{"severity":"error|warning","rule_id":"string","message":"string","target":"string"}]}'
        )
        user_prompt = (
            f"Diagram type: {diagram_type}\n"
            "Latest user intent:\n"
            f"{user_message}\n\n"
            "Current Mermaid code before update:\n"
            f"{current_code}\n\n"
            "Candidate Mermaid code:\n"
            f"{candidate_code}\n\n"
            "Rules:\n"
            "- Report only actionable findings.\n"
            "- severity must be error or warning.\n"
            "- Keep findings short.\n"
        )
        try:
            result = self.llm_client.complete_json(system_prompt, user_prompt, temperature=0.0)
        except Exception:
            return []
        raw_findings = result.get("findings", []) if isinstance(result, dict) else []
        findings: List[ValidationFinding] = []
        for raw in raw_findings:
            if not isinstance(raw, dict):
                continue
            severity = str(raw.get("severity", "warning")).strip().lower()
            if severity not in {"error", "warning"}:
                severity = "warning"
            findings.append(
                ValidationFinding(
                    severity=severity,
                    rule_id=str(raw.get("rule_id", "llm_check")).strip() or "llm_check",
                    message=str(raw.get("message", "")).strip() or "LLM suggested a potential issue.",
                    target=str(raw.get("target", "")).strip(),
                )
            )
        return findings

    def _deterministic_checks(self, diagram_type: str, candidate_code: str) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        stripped = (candidate_code or "").strip()
        if not stripped:
            return [ValidationFinding("error", "empty_code", "Mermaid code is empty.")]

        lines = [line.rstrip() for line in stripped.splitlines()]
        header = MERMAID_HEADERS.get(diagram_type, "")
        if header:
            first = lines[0].strip() if lines else ""
            if first != header:
                findings.append(
                    ValidationFinding(
                        "error",
                        "header_mismatch",
                        f"Expected header '{header}' but got '{first or '(none)'}'.",
                    )
                )

        for line in lines[1:]:
            foreign = _detect_foreign_header(line, expected_header=header)
            if foreign:
                findings.append(
                    ValidationFinding(
                        "error",
                        "diagram_type_mismatch",
                        f"Detected foreign diagram header '{foreign}' in {diagram_type} code.",
                    )
                )
                break

        body_lines = [line.strip() for line in lines[1:] if line.strip() and not line.strip().startswith("%%")]
        if not body_lines:
            findings.append(ValidationFinding("error", "missing_body", "Diagram body is empty."))
            return findings

        if diagram_type == "Sequence":
            findings.extend(self._validate_sequence(body_lines))
        elif diagram_type == "State":
            findings.extend(self._validate_state(body_lines))
        elif diagram_type == "Class":
            findings.extend(self._validate_class(body_lines))
        elif diagram_type == "ER":
            findings.extend(self._validate_er(body_lines))
        elif diagram_type == "Gantt":
            findings.extend(self._validate_gantt(body_lines))

        return findings

    def _validate_sequence(self, body_lines: List[str]) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        participant_re = re.compile(r"^\s*participant\s+([A-Za-z_][A-Za-z0-9_]*)\b")
        message_re = re.compile(
            r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[-.]+[<x>\-]*\s*([A-Za-z_][A-Za-z0-9_]*)\s*:"
        )

        participants = set()
        interactions = 0
        for line in body_lines:
            p = participant_re.match(line)
            if p:
                participants.add(p.group(1))
                continue
            m = message_re.match(line)
            if not m:
                continue
            interactions += 1
            src = m.group(1)
            dst = m.group(2)
            if participants and src not in participants:
                findings.append(
                    ValidationFinding("error", "unknown_participant", f"Unknown participant '{src}'.", target=src)
                )
            if participants and dst not in participants:
                findings.append(
                    ValidationFinding("error", "unknown_participant", f"Unknown participant '{dst}'.", target=dst)
                )
        if interactions == 0:
            findings.append(ValidationFinding("warning", "no_interaction", "No sequence interaction lines found."))
        return findings

    def _validate_state(self, body_lines: List[str]) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        transitions = [line for line in body_lines if "-->" in line]
        if not transitions:
            findings.append(ValidationFinding("warning", "no_transition", "No state transitions found."))
            return findings
        for line in transitions:
            left, right = line.split("-->", 1)
            src = left.strip()
            dst = right.split(":", 1)[0].strip()
            if not src:
                findings.append(ValidationFinding("error", "missing_source_state", "State transition source is empty."))
            if not dst:
                findings.append(ValidationFinding("error", "missing_target_state", "State transition target is empty."))
        return findings

    def _validate_class(self, body_lines: List[str]) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        inherit_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*<\|--\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")
        for line in body_lines:
            match = inherit_re.match(line)
            if not match:
                continue
            parent = match.group(1)
            child = match.group(2)
            if parent == child:
                findings.append(
                    ValidationFinding(
                        "error",
                        "self_inheritance",
                        f"Class '{child}' cannot inherit from itself.",
                        target=child,
                    )
                )
        return findings

    def _validate_er(self, body_lines: List[str]) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        entity_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*$")
        rel_re = re.compile(
            r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+[\|\}\{o]+--[\|\}\{o]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*:"
        )
        entities = set()
        for line in body_lines:
            match = entity_re.match(line)
            if match:
                entities.add(match.group(1))

        rel_count = 0
        for line in body_lines:
            match = rel_re.match(line)
            if not match:
                continue
            rel_count += 1
            left = match.group(1)
            right = match.group(2)
            if entities and left not in entities:
                findings.append(
                    ValidationFinding("warning", "unknown_entity", f"Relationship references unknown entity '{left}'.")
                )
            if entities and right not in entities:
                findings.append(
                    ValidationFinding("warning", "unknown_entity", f"Relationship references unknown entity '{right}'.")
                )
        if rel_count == 0:
            findings.append(ValidationFinding("warning", "no_relationship", "No ER relationship lines found."))
        return findings

    def _validate_gantt(self, body_lines: List[str]) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        task_id_re = re.compile(r":\s*(?:active|done|crit|milestone)?\s*,?\s*([A-Za-z_][A-Za-z0-9_-]*)\s*,")
        dep_re = re.compile(r"\bafter\s+([A-Za-z_][A-Za-z0-9_-]*)\b")

        task_ids = set()
        for line in body_lines:
            match = task_id_re.search(line)
            if match:
                task_ids.add(match.group(1))

        for line in body_lines:
            for dep in dep_re.findall(line):
                if dep not in task_ids:
                    findings.append(
                        ValidationFinding(
                            "warning",
                            "unknown_dependency_task",
                            f"Gantt dependency references unknown task id '{dep}'.",
                            target=dep,
                        )
                    )
        return findings


def _detect_foreign_header(line: str, expected_header: str) -> str:
    candidate = (line or "").strip()
    if not candidate:
        return ""
    for prefix in HEADER_PREFIXES:
        if prefix == "graph ":
            is_header = candidate.startswith("graph ")
            normalized = "graph TD" if is_header else ""
        else:
            is_header = candidate == prefix
            normalized = prefix
        if not is_header:
            continue
        if expected_header and normalized != expected_header:
            return normalized
    return ""
