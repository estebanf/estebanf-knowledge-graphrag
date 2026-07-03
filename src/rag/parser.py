import re
from dataclasses import dataclass
from pathlib import Path

import structlog

from rag.image_description import describe_image

log = structlog.get_logger()

_IMAGE_REF_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_REMOTE_PREFIXES = ("http://", "https://", "data:")
_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class ParseError(Exception):
    pass


@dataclass(frozen=True)
class ParseResult:
    markdown: str
    element_tree: str


_TXT_EXTENSIONS = {".txt", ".text"}
_MARKDOWN_EXTENSIONS = {".md", ".markdown"}


def _plaintext_to_element_tree(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    nonempty = [line for line in lines if line]

    tree_lines = ["0: document with name=_root_"]
    if nonempty and nonempty[0].startswith("#"):
        title = nonempty[0].lstrip("#").strip()
        tree_lines.append(f" 1: title: {title}")
        body = nonempty[1:]
    else:
        body = nonempty

    for index, line in enumerate(body, start=2):
        tree_lines.append(f" {index}: paragraph: {line}")

    return "\n".join(tree_lines)


def _describe_markdown_images(text: str, base_dir: Path) -> str:
    def _replace(m: re.Match) -> str:
        path_str = m.group(2)
        if path_str.startswith(_REMOTE_PREFIXES):
            return m.group(0)
        img_path = base_dir / path_str
        if not img_path.exists():
            return m.group(0)
        mime = _MIME_MAP.get(img_path.suffix.lower())
        if mime is None:
            return m.group(0)
        try:
            return describe_image(img_path.read_bytes(), mime)
        except Exception as exc:
            log.warning("image_description_skipped", path=str(img_path), error=str(exc))
            return m.group(0)

    return _IMAGE_REF_RE.sub(_replace, text)


def parse_document(file_path: Path) -> ParseResult:
    """Parse a stored source into markdown.

    Backend-safe by construction: only markdown/text are handled here, so the
    backend worker never needs Docling. Binary documents (PDF/DOCX/PPTX) are
    prepared into self-contained markdown on the CLI (see ``rag.prepare``) before
    a job is queued; if one ever reaches the worker, this raises a clear error
    rather than attempting a Docling conversion the backend cannot perform.
    """
    suffix = file_path.suffix.lower()
    if suffix in _TXT_EXTENSIONS | _MARKDOWN_EXTENSIONS:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            if suffix in _MARKDOWN_EXTENSIONS:
                text = _describe_markdown_images(text, file_path.parent)
            return ParseResult(markdown=text, element_tree=_plaintext_to_element_tree(text))
        except Exception as exc:
            raise ParseError(f"Failed to parse {file_path.name}: {exc}") from exc

    raise ParseError(
        f"Unsupported binary document '{file_path.name}' ({suffix or 'unknown'}): "
        "backend workers parse markdown/text only. Prepare binary documents on the "
        "CLI with `rag prepare` or `rag ingest` before submitting them."
    )


def parse_to_markdown(file_path: Path) -> str:
    return parse_document(file_path).markdown
