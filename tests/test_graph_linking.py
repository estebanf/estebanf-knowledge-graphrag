from unittest.mock import patch, MagicMock


def test_find_dedup_candidates_uses_sql_similarity():
    """find_dedup_candidates must query DB for similarity, not call embedding API."""
    conn = MagicMock()
    # SQL returns (src_id, ex_id, similarity) rows already above threshold
    conn.execute.return_value.fetchall.return_value = [
        ("src-id-1", "ex-id-1", 0.97),
    ]

    from rag.graph_linking import find_dedup_candidates
    pairs = find_dedup_candidates(conn, "source-uuid")

    assert len(pairs) == 1
    assert pairs[0] == ("src-id-1", "ex-id-1", 0.97)

    # Verify no embedding API is called from graph_linking
    with patch("rag.graph_linking.get_embeddings", create=True) as mock_embed:
        find_dedup_candidates(conn, "source-uuid")
    mock_embed.assert_not_called()


def test_find_dedup_candidates_returns_empty_when_no_matches():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []

    from rag.graph_linking import find_dedup_candidates
    result = find_dedup_candidates(conn, "source-uuid")
    assert result == []


def test_create_mentioned_in_edges_uses_single_unwind_query():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("chunk-1",), ("chunk-2",),
    ]
    session_mock = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session_mock)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    from rag.graph_linking import create_mentioned_in_edges
    create_mentioned_in_edges(conn, driver, "source-uuid")

    # exactly one Cypher call with UNWIND
    assert session_mock.run.call_count == 1
    cypher = session_mock.run.call_args[0][0]
    kwargs = session_mock.run.call_args[1]
    assert "UNWIND" in cypher
    assert "MENTIONED_IN" in cypher
    assert kwargs["chunk_ids"] == ["chunk-1", "chunk-2"]


def test_create_mentioned_in_edges_no_chunks_skips_memgraph():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    driver = MagicMock()

    from rag.graph_linking import create_mentioned_in_edges
    create_mentioned_in_edges(conn, driver, "source-uuid")

    driver.session.assert_not_called()


def test_link_graph_calls_dedup_and_linking():
    conn = MagicMock()
    driver = MagicMock()
    with patch("rag.graph_linking.find_dedup_candidates", return_value=[]) as mock_dedup, \
         patch("rag.graph_linking.create_mentioned_in_edges") as mock_link:
        from rag.graph_linking import link_graph
        link_graph(conn, driver, "source-uuid", "job-uuid")
    mock_dedup.assert_called_once()
    mock_link.assert_called_once()
