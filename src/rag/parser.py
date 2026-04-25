import io
import re
from dataclasses import dataclass
from pathlib import Path

from rag.image_description import describe_image

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

_converter = None


def _get_docling_converter():
    global _converter
    if _converter is not None:
        return _converter

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = True
    pdf_options.do_table_structure = True
    pdf_options.generate_page_images = False
    pdf_options.generate_picture_images = True

    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        }
    )
    return _converter


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
        mime = _MIME_MAP.get(img_path.suffix.lower(), "image/png")
        return describe_image(img_path.read_bytes(), mime)

    return _IMAGE_REF_RE.sub(_replace, text)


def _describe_docling_pictures(result, markdown: str) -> str:
    for picture in result.document.pictures:
        img = picture.get_image(result.document)
        if img is None:
            continue
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        description = describe_image(buf.getvalue(), "image/png")
        markdown = markdown.replace("<!-- image -->", description, 1)
    return markdown


def parse_document(file_path: Path) -> ParseResult:
    try:
        suffix = file_path.suffix.lower()
        if suffix in _TXT_EXTENSIONS | _MARKDOWN_EXTENSIONS:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            if suffix in _MARKDOWN_EXTENSIONS:
                text = _describe_markdown_images(text, file_path.parent)
            return ParseResult(markdown=text, element_tree=_plaintext_to_element_tree(text))

        result = _get_docling_converter().convert(str(file_path))
        markdown = result.document.export_to_markdown()
        markdown = _describe_docling_pictures(result, markdown)
        return ParseResult(
            markdown=markdown,
            element_tree=result.document.export_to_element_tree(),
        )
    except Exception as exc:
        raise ParseError(f"Failed to parse {file_path.name}: {exc}") from exc


def parse_to_markdown(file_path: Path) -> str:
    return parse_document(file_path).markdown
