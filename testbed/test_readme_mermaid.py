from pathlib import Path
import re


RESERVED_IDS = {
    "end",
}


def _extract_mermaid_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    pattern = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    for match in pattern.finditer(text):
        blocks.append(match.group(1))
    return blocks


def test_readme_mermaid_node_ids_avoid_reserved_tokens():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    content = readme.read_text(encoding="utf-8")
    blocks = _extract_mermaid_blocks(content)
    assert blocks, "README.md must contain at least one Mermaid block"

    declared_ids: list[str] = []
    node_decl_pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[")
    for block in blocks:
        for line in block.splitlines():
            match = node_decl_pattern.match(line.strip())
            if not match:
                continue
            declared_ids.append(match.group(1))

    offending = sorted({node_id for node_id in declared_ids if node_id.lower() in RESERVED_IDS})
    assert not offending, f"Reserved Mermaid node IDs found in README: {offending}"
