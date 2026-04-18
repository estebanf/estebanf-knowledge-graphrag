from unittest.mock import patch, MagicMock
from rag.profiling import DocumentProfile
from rag.chunking import chunk_document, ChunkData, select_strategy

WELL_STRUCTURED_MD = """# Chapter 1

This is the first section with some content here that discusses various topics.

## Section 1.1

More detailed content in this subsection about the first topic.

# Chapter 2

Second chapter content with additional material.

## Section 2.1

Another subsection with content.
"""

PROSE_MD = """This is a long prose document with no headings. It contains multiple paragraphs of text that flow together without clear structural markers. The content is dense and uniform throughout. This is a long prose document with no headings. It contains multiple paragraphs of text that flow together without clear structural markers. The content is dense and uniform throughout."""

LEGAL_MD = """WHEREAS, the parties agree to the following terms and conditions. The obligations set forth herein are binding on all signatories. Any breach of these terms shall result in damages as specified below. The governing law shall be that of the applicable jurisdiction."""


def _make_profile(**kwargs):
    defaults = dict(
        structure_type="unstructured",
        heading_consistency="none",
        content_density="uniform",
        primary_content_type="prose",
        avg_section_length="medium",
        has_tables=False,
        has_code_blocks=False,
        domain="general",
    )
    defaults.update(kwargs)
    return DocumentProfile(**defaults)


def test_select_strategy_well_structured():
    profile = _make_profile(structure_type="well-structured", heading_consistency="consistent")
    assert select_strategy(profile) == "markdown-header"


def test_select_strategy_transcript():
    profile = _make_profile(primary_content_type="transcript")
    assert select_strategy(profile) == "semantic"


def test_select_strategy_fallback():
    profile = _make_profile(structure_type="loosely-structured")
    assert select_strategy(profile) == "semantic"


def test_chunk_document_returns_list_of_chunkdata():
    profile = _make_profile(structure_type="well-structured", heading_consistency="consistent")
    chunks = chunk_document(WELL_STRUCTURED_MD, profile)
    assert len(chunks) > 0
    assert all(isinstance(c, ChunkData) for c in chunks)


def test_chunk_document_sets_chunking_strategy():
    profile = _make_profile()
    chunks = chunk_document(PROSE_MD, profile)
    assert all(c.chunking_strategy == "recursive" for c in chunks)


def test_chunk_document_sets_chunk_index():
    profile = _make_profile()
    chunks = chunk_document(PROSE_MD, profile)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_document_token_count_positive():
    profile = _make_profile()
    chunks = chunk_document(PROSE_MD, profile)
    assert all(c.token_count > 0 for c in chunks)


def test_chunk_document_no_api_key_skips_propositions(monkeypatch):
    monkeypatch.setattr("rag.chunking.settings.OPENROUTER_API_KEY", "")
    profile = _make_profile(domain="legal")
    chunks = chunk_document(LEGAL_MD, profile)
    assert len(chunks) > 0
    assert all(c.chunking_strategy != "proposition" for c in chunks)


def test_proposition_chunking_called_for_legal_domain(monkeypatch):
    monkeypatch.setattr("rag.chunking.settings.OPENROUTER_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '["Fact one.", "Fact two.", "Fact three."]'}}]
    }
    with patch("rag.chunking.requests.post", return_value=mock_resp):
        profile = _make_profile(domain="legal")
        chunks = chunk_document(LEGAL_MD, profile)
    proposition_chunks = [c for c in chunks if c.chunking_strategy == "proposition"]
    assert len(proposition_chunks) > 0


def test_proposition_chunks_have_parent_chunk_id(monkeypatch):
    monkeypatch.setattr("rag.chunking.settings.OPENROUTER_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '["Fact one.", "Fact two."]'}}]
    }
    with patch("rag.chunking.requests.post", return_value=mock_resp):
        profile = _make_profile(domain="legal")
        chunks = chunk_document(LEGAL_MD, profile)
    proposition_chunks = [c for c in chunks if c.chunking_strategy == "proposition"]
    assert all(c.parent_chunk_id is not None for c in proposition_chunks)
    assert all(c.parent_chunk_id.isdigit() for c in proposition_chunks)


def test_hierarchical_chunking_preserves_base_strategy_metadata():
    profile = _make_profile(
        structure_type="well-structured",
        heading_consistency="consistent",
        avg_section_length="long",
    )
    chunks = chunk_document(WELL_STRUCTURED_MD * 30, profile)

    parent_chunks = [c for c in chunks if c.parent_chunk_id is None]
    child_chunks = [c for c in chunks if c.parent_chunk_id is not None]

    assert parent_chunks
    assert child_chunks
    assert all(c.chunking_strategy == "markdown-header" for c in parent_chunks)
    assert all(c.metadata.get("base_strategy") == "markdown-header" for c in child_chunks)
