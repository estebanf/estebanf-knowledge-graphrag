import io
from unittest.mock import patch

import pytest
from PIL import Image as PILImage

from rag.parser import ParseError, ParseResult, parse_document


def test_parse_document_returns_structured_result_for_markdown(tmp_path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Title\n\nParagraph.\n", encoding="utf-8")

    result = parse_document(file_path)

    assert isinstance(result, ParseResult)
    assert result.markdown.startswith("# Title")
    assert "paragraph" in result.element_tree.lower()


def test_parse_document_parses_markdown_without_docling(tmp_path):
    # Backend-safe: markdown parsing never touches Docling.
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Title\n\nParagraph.\n", encoding="utf-8")

    result = parse_document(file_path)

    assert result.markdown.startswith("# Title")
    import sys

    assert not any(name.startswith("docling") for name in sys.modules)


@patch("rag.parser.describe_image", return_value="A red square.")
def test_parse_document_replaces_local_image_refs_in_markdown(mock_describe, tmp_path):
    img = PILImage.new("RGB", (10, 10), color="red")
    img_path = tmp_path / "chart.png"
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_path.write_bytes(buf.getvalue())

    md_file = tmp_path / "report.md"
    md_file.write_text("# Report\n\n![chart](chart.png)\n", encoding="utf-8")

    result = parse_document(md_file)

    assert "![chart](chart.png)" not in result.markdown
    assert "A red square." in result.markdown
    mock_describe.assert_called_once()


@patch("rag.parser.describe_image")
def test_parse_document_leaves_remote_image_refs_in_markdown(mock_describe, tmp_path):
    md_file = tmp_path / "report.md"
    md_file.write_text("![x](https://example.com/img.png)\n", encoding="utf-8")

    result = parse_document(md_file)

    mock_describe.assert_not_called()
    assert "https://example.com/img.png" in result.markdown


@patch("rag.parser.describe_image")
def test_parse_document_skips_missing_local_images_in_markdown(mock_describe, tmp_path):
    md_file = tmp_path / "report.md"
    md_file.write_text("![x](missing.png)\n", encoding="utf-8")

    result = parse_document(md_file)

    mock_describe.assert_not_called()


def test_parse_document_raises_clear_error_for_binary_formats(tmp_path):
    # The backend worker parses markdown/text only. A binary suffix must raise a
    # clear error immediately, without attempting (or importing) Docling.
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    with pytest.raises(ParseError, match="markdown/text only"):
        parse_document(file_path)

    import sys

    assert not any(name.startswith("docling") for name in sys.modules)
