from unittest.mock import MagicMock, patch

from rag.retrieval import HybridSearchResults, InsightSearchResult, RetrievalCandidate, _expand_chunk_texts, hybrid_search, sparse_retrieve


def _candidate(chunk_id: str, score: float = 0.8) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        chunk=f"chunk content {chunk_id}",
        source_id="source-1",
        source_path="/tmp/doc.md",
        source_metadata={},
        score=score,
    )


def _empty_insights() -> list[InsightSearchResult]:
    return []


@patch("rag.retrieval.insight_hybrid_search")
@patch("rag.retrieval.sparse_retrieve")
@patch("rag.retrieval.dense_retrieve")
@patch("rag.retrieval.get_connection")
@patch("rag.retrieval._expand_chunk_texts")
def test_hybrid_search_returns_fused_results(mock_expand, mock_conn, mock_dense, mock_sparse, mock_insight):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    mock_dense.return_value = [_candidate("chunk-1"), _candidate("chunk-2")]
    mock_sparse.return_value = [_candidate("chunk-2"), _candidate("chunk-3")]
    mock_insight.return_value = _empty_insights()
    mock_expand.return_value = {
        "chunk-1": "expanded chunk-1",
        "chunk-2": "expanded chunk-2",
        "chunk-3": "expanded chunk-3",
    }

    results = hybrid_search("test query", limit=10, min_score=0.0)

    assert isinstance(results, HybridSearchResults)
    assert isinstance(results.chunks, list)
    assert all(isinstance(r, RetrievalCandidate) for r in results.chunks)
    chunk_ids = [r.chunk_id for r in results.chunks]
    assert "chunk-1" in chunk_ids
    assert "chunk-2" in chunk_ids
    assert "chunk-3" in chunk_ids
    assert results.chunks[0].chunk.startswith("expanded")
    mock_dense.assert_called_once()
    mock_sparse.assert_called_once()


@patch("rag.retrieval.insight_hybrid_search")
@patch("rag.retrieval.sparse_retrieve")
@patch("rag.retrieval.dense_retrieve")
@patch("rag.retrieval.get_connection")
@patch("rag.retrieval._expand_chunk_texts", return_value={})
def test_hybrid_search_respects_limit(mock_expand, mock_conn, mock_dense, mock_sparse, mock_insight):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    mock_dense.return_value = [_candidate(f"chunk-{i}") for i in range(10)]
    mock_sparse.return_value = [_candidate(f"chunk-{i}") for i in range(5, 15)]
    mock_insight.return_value = _empty_insights()

    results = hybrid_search("test query", limit=3, min_score=0.0)

    assert len(results.chunks) <= 3


@patch("rag.retrieval.insight_hybrid_search")
@patch("rag.retrieval.sparse_retrieve")
@patch("rag.retrieval.dense_retrieve")
@patch("rag.retrieval.get_connection")
@patch("rag.retrieval._expand_chunk_texts", return_value={})
def test_hybrid_search_filters_by_cosine_similarity(mock_expand, mock_conn, mock_dense, mock_sparse, mock_insight):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    mock_dense.return_value = [_candidate("chunk-1", score=0.5), _candidate("chunk-2", score=0.9)]
    mock_sparse.return_value = []
    mock_insight.return_value = _empty_insights()

    results = hybrid_search("test query", limit=10, min_score=0.7)

    assert len(results.chunks) == 1
    assert results.chunks[0].chunk_id == "chunk-2"
    assert results.chunks[0].score == 0.9


@patch("rag.retrieval.insight_hybrid_search")
@patch("rag.retrieval.sparse_retrieve")
@patch("rag.retrieval.dense_retrieve")
@patch("rag.retrieval.get_connection")
@patch("rag.retrieval._expand_chunk_texts", return_value={})
def test_hybrid_search_score_is_cosine_similarity_not_rrf(mock_expand, mock_conn, mock_dense, mock_sparse, mock_insight):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    mock_dense.return_value = [_candidate("chunk-1", score=0.85)]
    mock_sparse.return_value = []
    mock_insight.return_value = _empty_insights()

    results = hybrid_search("test query", limit=10, min_score=0.0)

    assert len(results.chunks) == 1
    assert results.chunks[0].score == 0.85


@patch("rag.retrieval.get_embeddings")
@patch("rag.retrieval.insight_hybrid_search")
@patch("rag.retrieval.sparse_retrieve")
@patch("rag.retrieval.dense_retrieve")
@patch("rag.retrieval.get_connection")
@patch("rag.retrieval._expand_chunk_texts", return_value={})
def test_hybrid_search_returns_insights(mock_expand, mock_conn, mock_dense, mock_sparse, mock_insight, mock_embeddings):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    mock_embeddings.return_value = [[0.1] * 4096]
    mock_dense.return_value = [_candidate("chunk-1")]
    mock_sparse.return_value = []
    mock_insight.return_value = [
        InsightSearchResult(
            score=0.92,
            insight="An important insight",
            insight_id="insight-1",
            topics=["economics"],
            sources=[],
        )
    ]

    results = hybrid_search("test query", limit=10, min_score=0.0)

    assert isinstance(results, HybridSearchResults)
    assert len(results.chunks) == 1
    assert len(results.insights) == 1
    assert results.insights[0].insight == "An important insight"
    assert results.insights[0].topics == ["economics"]


def test_sparse_retrieve_uses_indexable_english_tsvector_expression():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []

    sparse_retrieve(conn, "insurance claim triage", source_ids=[], filters={}, top_n=30)

    sql = conn.execute.call_args.args[0]
    assert "to_tsvector('english', coalesce(c.content, ''))" in sql
    assert "websearch_to_tsquery('english', %s)" in sql
    assert "to_tsvector(%s" not in sql


def test_expand_chunk_texts_uses_uuid_ids_without_text_cast():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []

    _expand_chunk_texts(conn, ["00000000-0000-0000-0000-000000000001"])

    sql = conn.execute.call_args.args[0]
    params = conn.execute.call_args.args[1]
    assert "WHERE id = ANY(%s::uuid[])" in sql
    assert "WHERE id::text = ANY(%s::text[])" not in sql
    assert params[0] == ["00000000-0000-0000-0000-000000000001"]
