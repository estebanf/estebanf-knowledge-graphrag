import io
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image as PILImage

from rag.parser import ParseResult, parse_document


TEST_DOCS = Path(__file__).parent.parent / "test_documents"


def test_parse_document_returns_structured_result_for_markdown(tmp_path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Title\n\nParagraph.\n", encoding="utf-8")

    result = parse_document(file_path)

    assert isinstance(result, ParseResult)
    assert result.markdown.startswith("# Title")
    assert "paragraph" in result.element_tree.lower()


@patch("rag.parser.describe_image", return_value="[image]")
def test_parse_document_supports_pptx(mock_describe):
    file_path = TEST_DOCS / "From AI ambition to a decision-ready roadmap.pptx"

    result = parse_document(file_path)

    assert result.markdown
    assert "decision-ready roadmap" in result.markdown.lower()
    assert "slide-0" in result.element_tree.lower()


@patch("rag.parser.describe_image", return_value="A diagram showing a workflow.")
def test_parse_document_replaces_image_placeholders_for_pptx(mock_describe):
    file_path = TEST_DOCS / "From AI ambition to a decision-ready roadmap.pptx"

    result = parse_document(file_path)

    assert "<!-- image -->" not in result.markdown
    if mock_describe.called:
        assert "A diagram showing a workflow." in result.markdown


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


@patch("rag.parser._get_docling_converter", side_effect=RuntimeError("docling unavailable"))
def test_parse_document_does_not_require_docling_for_markdown(mock_converter, tmp_path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Title\n\nParagraph.\n", encoding="utf-8")

    result = parse_document(file_path)

    assert result.markdown.startswith("# Title")
    mock_converter.assert_not_called()


@patch("rag.parser._get_docling_converter", side_effect=RuntimeError("docling unavailable"))
def test_parse_document_raises_clear_error_for_binary_formats_without_docling(mock_converter, tmp_path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    with patch("rag.parser.describe_image"):
        from rag.parser import ParseError
        with pytest.raises(ParseError, match="docling unavailable"):
            parse_document(file_path)
