from src.mermaid_generator.attachment_prompt import (
    build_attachment_generation_prompt,
)


def test_build_attachment_generation_prompt_includes_attachment_text():
    prompt = build_attachment_generation_prompt(
        attachment_text="Step1\nStep2",
        diagram_type="Flowchart",
    )
    assert "Flowchart" in prompt
    assert "Step1" in prompt
    assert "Attachment Content" in prompt


def test_build_attachment_generation_prompt_truncates_long_text():
    long_text = "x" * 120
    prompt = build_attachment_generation_prompt(
        attachment_text=long_text,
        diagram_type="Sequence",
        max_chars=32,
    )
    assert len(prompt) < 400
    assert "...(truncated)" in prompt
