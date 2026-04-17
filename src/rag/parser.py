from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


class ParseError(Exception):
    pass


_TXT_EXTENSIONS = {".txt", ".text"}

_pdf_options = PdfPipelineOptions()
_pdf_options.do_ocr = False
_pdf_options.generate_page_images = False
_pdf_options.generate_picture_images = False

_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_options),
    }
)


def parse_to_markdown(file_path: Path) -> str:
    try:
        if file_path.suffix.lower() in _TXT_EXTENSIONS:
            return file_path.read_text(encoding="utf-8", errors="replace")

        result = _converter.convert(str(file_path))
        return result.document.export_to_markdown()
    except Exception as exc:
        raise ParseError(f"Failed to parse {file_path.name}: {exc}") from exc
