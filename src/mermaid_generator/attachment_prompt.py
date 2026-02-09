def build_attachment_generation_prompt(
    attachment_text: str,
    diagram_type: str,
    max_chars: int = 6000,
) -> str:
    text = (attachment_text or "").strip()
    if len(text) > max_chars:
        text = f"{text[:max_chars].rstrip()}\n...(truncated)"

    return (
        f"Create a {diagram_type} diagram from the attached content.\n"
        "Preserve key steps and decisions, and output a practical structure.\n\n"
        "Attachment Content:\n"
        f"{text}\n"
    )
