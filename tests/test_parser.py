from pathlib import Path
from unittest.mock import patch

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
