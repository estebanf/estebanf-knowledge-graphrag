"""CLI-side preparation of binary documents (PDF/DOCX/PPTX) into markdown.

This module carries the heavy Docling dependency chain (Torch/CUDA) and is only
installed on machines that ingest binary documents (the ``prepare`` extra). The
backend never imports it: workers process markdown/text only (see
``rag.parser.parse_document``), which is what lets the backend image drop Docling.

Preparation is split into two steps so the CLI can keep LLM credentials on the
backend (KTD2/R3):

- ``prepare_binary`` converts a binary document to markdown plus the raw bytes of
  each embedded image, leaving Docling's ``<!-- image -->`` placeholders in place.
- ``finalize_markdown`` replaces those placeholders with descriptions produced by
  an injected ``describe`` callback (the CLI routes it through the backend image
  API). It hard-fails if any placeholder cannot be resolved so a job is never
  queued with unresolved images (AE2, R2).

Docling is imported lazily so ``import rag.prepare`` stays cheap and Docling-free.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_IMAGE_PLACEHOLDER = "<!-- image -->"

# Callback the caller supplies to turn image bytes into a text description.
DescribeFn = Callable[[bytes, str], str]


class PrepareError(Exception):
    """Raised when a binary document cannot be prepared into ingestible markdown."""


@dataclass(frozen=True)
class ExtractedImage:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class PreparedDocument:
    """Result of converting a binary document, before image descriptions."""

    markdown: str
    images: list[ExtractedImage]
    original_filename: str
    original_extension: str
    original_md5: str
    image_count: int


_converter = None


def _get_docling_converter():
    global _converter
    if _converter is not None:
        return _converter

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:  # pragma: no cover - exercised via patched import in tests
        raise PrepareError(
            "Binary document preparation requires the optional 'prepare' extra "
            "(Docling). Install it with `pip install -e .[prepare]`."
        ) from exc

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


def _compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_binary(file_path: Path) -> PreparedDocument:
    """Convert a PDF/DOCX/PPTX into markdown plus the bytes of each embedded image.

    The returned markdown still contains Docling's ``<!-- image -->`` placeholders;
    call :func:`finalize_markdown` to substitute image descriptions. Raises
    :class:`PrepareError` when Docling is unavailable or conversion fails.
    """
    file_path = Path(file_path)
    try:
        result = _get_docling_converter().convert(str(file_path))
        markdown = result.document.export_to_markdown()
        images: list[ExtractedImage] = []
        for picture in result.document.pictures:
            img = picture.get_image(result.document)
            if img is None:
                continue
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append(ExtractedImage(data=buf.getvalue(), mime_type="image/png"))
    except PrepareError:
        raise
    except Exception as exc:
        raise PrepareError(f"Failed to prepare {file_path.name}: {exc}") from exc

    return PreparedDocument(
        markdown=markdown,
        images=images,
        original_filename=file_path.name,
        original_extension=file_path.suffix.lower().lstrip("."),
        original_md5=_compute_md5(file_path),
        image_count=len(images),
    )


def finalize_markdown(prepared: PreparedDocument, describe: DescribeFn) -> str:
    """Replace each ``<!-- image -->`` placeholder with a description, in order.

    ``describe`` is called once per extracted image with ``(bytes, mime_type)``.
    Hard-fails (raises :class:`PrepareError`) if a description raises or if any
    placeholder remains unresolved, so a prepared document is never submitted for
    ingestion with dangling image markers (AE2).
    """
    markdown = prepared.markdown
    for index, image in enumerate(prepared.images, start=1):
        try:
            description = describe(image.data, image.mime_type)
        except Exception as exc:
            raise PrepareError(
                f"Failed to describe image {index} of {prepared.original_filename}: {exc}"
            ) from exc
        markdown = markdown.replace(_IMAGE_PLACEHOLDER, description, 1)

    if _IMAGE_PLACEHOLDER in markdown:
        raise PrepareError(
            f"Unresolved image placeholders remain in prepared markdown for "
            f"{prepared.original_filename}: extracted {prepared.image_count} image(s) "
            "but more placeholders were present."
        )
    return markdown
