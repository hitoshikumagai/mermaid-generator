import pytest

from src.mermaid_generator.attachment_text import (
    EmptyAttachmentError,
    UnsupportedAttachmentError,
    extract_text_from_attachment,
)


def test_extract_text_from_txt_utf8():
    text = extract_text_from_attachment("notes.txt", "hello\nworld".encode("utf-8"))
    assert text == "hello\nworld"


def test_extract_text_from_md_utf8():
    text = extract_text_from_attachment("plan.md", "# title\n- item".encode("utf-8"))
    assert text.startswith("# title")


def test_extract_text_from_txt_latin1_fallback():
    raw = "cafe\xe9".encode("latin-1")
    text = extract_text_from_attachment("memo.txt", raw)
    assert "cafe" in text


def test_extract_raises_for_unsupported_extension():
    with pytest.raises(UnsupportedAttachmentError):
        extract_text_from_attachment("spec.pdf", b"%PDF-1.7")


def test_extract_raises_for_empty_attachment():
    with pytest.raises(EmptyAttachmentError):
        extract_text_from_attachment("empty.md", b"")
