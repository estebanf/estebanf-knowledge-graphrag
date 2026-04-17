import pytest
from unittest.mock import patch, MagicMock


def _make_embedding(seed: float, dim: int = 4096) -> list[float]:
    # Use seed as first element, 0 elsewhere → orthogonal unit vectors
    vec = [seed] + [0.0] * (dim - 1)
    norm = abs(seed) if seed != 0 else 1.0
    return [x / norm for x in vec]


def test_embed_entity_names_delegates_to_get_embeddings():
    vectors = [[0.1] * 4096, [0.2] * 4096]
    with patch("rag.graph_linking.get_embeddings", return_value=vectors) as mock_embed:
        from rag.graph_linking import embed_entity_names
        result = embed_entity_names(["Acme", "Bob"])
    mock_embed.assert_called_once_with(["Acme", "Bob"])
    assert result == vectors


def test_embed_entity_names_empty_returns_empty():
    with patch("rag.graph_linking.get_embeddings") as mock_embed:
        from rag.graph_linking import embed_entity_names
        result = embed_entity_names([])
    assert result == []
    mock_embed.assert_not_called()


def test_find_dedup_candidates_returns_high_similarity_pairs(monkeypatch):
    monkeypatch.setattr("rag.graph_linking.settings.ENTITY_DEDUP_COSINE_THRESHOLD", 0.92)

    vec_a = _make_embedding(1.0)
    vec_b = _make_embedding(1.0)    # identical → cosine = 1.0
    # orthogonal vector: second element is 1, first is 0
    vec_c = [0.0] + [1.0] + [0.0] * (4096 - 2)  # cosine with vec_a = 0.0

    conn = MagicMock()
    conn.execute.return_value.fetchall.side_effect = [
        [("src-id-1", "Acme Corp")],
        [("ex-id-1", "Acme Corporation"), ("ex-id-2", "Widgets")],
    ]

    with patch("rag.graph_linking.embed_entity_names", side_effect=[
        [vec_a], [vec_b, vec_c]
    ]):
        from rag.graph_linking import find_dedup_candidates
        pairs = find_dedup_candidates(conn, "source-uuid")

    assert len(pairs) == 1
    assert pairs[0][0] == "src-id-1"
    assert pairs[0][1] == "ex-id-1"


def test_find_dedup_candidates_no_existing_returns_empty():
    conn = MagicMock()
    conn.execute.return_value.fetchall.side_effect = [
        [("src-id-1", "Acme")],
        [],
    ]
    with patch("rag.graph_linking.embed_entity_names", return_value=[[0.1] * 4096]):
        from rag.graph_linking import find_dedup_candidates
        result = find_dedup_candidates(conn, "source-uuid")
    assert result == []


def test_create_mentioned_in_edges_creates_memgraph_edges():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("entity-1", "chunk-1"),
        ("entity-1", "chunk-2"),
    ]
    session_mock = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session_mock)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    from rag.graph_linking import create_mentioned_in_edges
    create_mentioned_in_edges(conn, driver, "source-uuid")

    assert session_mock.run.call_count == 2
    cypher_calls = [str(c) for c in session_mock.run.call_args_list]
    assert all("MENTIONED_IN" in c for c in cypher_calls)


def test_link_graph_calls_dedup_and_linking():
    conn = MagicMock()
    driver = MagicMock()
    with patch("rag.graph_linking.find_dedup_candidates", return_value=[]) as mock_dedup, \
         patch("rag.graph_linking.create_mentioned_in_edges") as mock_link:
        from rag.graph_linking import link_graph
        link_graph(conn, driver, "source-uuid", "job-uuid")
    mock_dedup.assert_called_once()
    mock_link.assert_called_once()
