from pathlib import Path


SUPPORTED_ATTACHMENT_EXTENSIONS = {".txt", ".md", ".markdown"}


class AttachmentError(ValueError):
    pass


class UnsupportedAttachmentError(AttachmentError):
    pass


class EmptyAttachmentError(AttachmentError):
    pass


def extract_text_from_attachment(filename: str, content_bytes: bytes) -> str:
    extension = Path((filename or "").strip()).suffix.lower()
    if extension not in SUPPORTED_ATTACHMENT_EXTENSIONS:
        raise UnsupportedAttachmentError(
            f"Unsupported attachment type: {extension or '(no extension)'}"
        )

    if not content_bytes:
        raise EmptyAttachmentError("Attachment is empty.")

    text = _decode_bytes(content_bytes).strip()
    if not text:
        raise EmptyAttachmentError("Attachment has no readable text.")
    return text


def _decode_bytes(content_bytes: bytes) -> str:
    encodings = ("utf-8-sig", "utf-8", "cp932", "latin-1")
    for encoding in encodings:
        try:
            return content_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content_bytes.decode("utf-8", errors="replace")
