from unittest.mock import MagicMock, patch

from rag.retrieval import RetrievalCandidate, hybrid_search


def _candidate(chunk_id: str, score: float = 0.8) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        chunk=f"chunk content {chunk_id}",
        source_id="source-1",
        source_path="/tmp/doc.md",
        source_metadata={},
        score=score,
    )


@patch("rag.retrieval.sparse_retrieve")
@patch("rag.retrieval.dense_retrieve")
@patch("rag.retrieval.get_connection")
def test_hybrid_search_returns_fused_results(mock_conn, mock_dense, mock_sparse):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    mock_dense.return_value = [_candidate("chunk-1"), _candidate("chunk-2")]
    mock_sparse.return_value = [_candidate("chunk-2"), _candidate("chunk-3")]

    results = hybrid_search("test query", limit=10, min_score=0.0)

    assert isinstance(results, list)
    assert all(isinstance(r, RetrievalCandidate) for r in results)
    chunk_ids = [r.chunk_id for r in results]
    assert "chunk-1" in chunk_ids
    assert "chunk-2" in chunk_ids
    assert "chunk-3" in chunk_ids
    mock_dense.assert_called_once()
    mock_sparse.assert_called_once()


@patch("rag.retrieval.sparse_retrieve")
@patch("rag.retrieval.dense_retrieve")
@patch("rag.retrieval.get_connection")
def test_hybrid_search_respects_limit(mock_conn, mock_dense, mock_sparse):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    mock_dense.return_value = [_candidate(f"chunk-{i}") for i in range(10)]
    mock_sparse.return_value = [_candidate(f"chunk-{i}") for i in range(5, 15)]

    results = hybrid_search("test query", limit=3, min_score=0.0)

    assert len(results) <= 3


@patch("rag.retrieval.sparse_retrieve")
@patch("rag.retrieval.dense_retrieve")
@patch("rag.retrieval.get_connection")
def test_hybrid_search_filters_by_cosine_similarity(mock_conn, mock_dense, mock_sparse):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    mock_dense.return_value = [_candidate("chunk-1", score=0.5), _candidate("chunk-2", score=0.9)]
    mock_sparse.return_value = []

    results = hybrid_search("test query", limit=10, min_score=0.7)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-2"
    assert results[0].score == 0.9


@patch("rag.retrieval.sparse_retrieve")
@patch("rag.retrieval.dense_retrieve")
@patch("rag.retrieval.get_connection")
def test_hybrid_search_score_is_cosine_similarity_not_rrf(mock_conn, mock_dense, mock_sparse):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    mock_dense.return_value = [_candidate("chunk-1", score=0.85)]
    mock_sparse.return_value = []

    results = hybrid_search("test query", limit=10, min_score=0.0)

    assert len(results) == 1
    assert results[0].score == 0.85
