from pathlib import Path

from rag.parser import ParseResult, parse_document


TEST_DOCS = Path(__file__).parent.parent / "test_documents"


def test_parse_document_returns_structured_result_for_markdown(tmp_path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Title\n\nParagraph.\n", encoding="utf-8")

    result = parse_document(file_path)

    assert isinstance(result, ParseResult)
    assert result.markdown.startswith("# Title")
    assert "paragraph" in result.element_tree.lower()


def test_parse_document_supports_pptx():
    file_path = TEST_DOCS / "From AI ambition to a decision-ready roadmap.pptx"

    result = parse_document(file_path)

    assert result.markdown
    assert "decision-ready roadmap" in result.markdown.lower()
    assert "slide-0" in result.element_tree.lower()
