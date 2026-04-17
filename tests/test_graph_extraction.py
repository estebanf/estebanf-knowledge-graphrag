import json
from unittest.mock import patch, MagicMock
import pytest


def _mock_llm_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"choices": [{"message": {"content": content}}]}
    return mock


def test_extract_entities_returns_list():
    payload = json.dumps([
        {"canonical_name": "Acme Corp", "entity_type": "ORGANIZATION", "aliases": ["Acme"]}
    ])
    with patch("rag.graph_extraction.requests.post", return_value=_mock_llm_response(payload)):
        from rag.graph_extraction import extract_entities
        result = extract_entities("Acme Corp acquired Widgets Inc last year.")
    assert len(result) == 1
    assert result[0]["canonical_name"] == "Acme Corp"
    assert result[0]["entity_type"] == "ORGANIZATION"


def test_extract_entities_returns_empty_on_api_error():
    with patch("rag.graph_extraction.requests.post", side_effect=Exception("network error")):
        from rag.graph_extraction import extract_entities
        result = extract_entities("Some text.")
    assert result == []


def test_extract_entities_returns_empty_without_api_key(monkeypatch):
    monkeypatch.setattr("rag.graph_extraction.settings.OPENROUTER_API_KEY", "")
    from rag.graph_extraction import extract_entities
    with patch("rag.graph_extraction.requests.post") as mock_post:
        result = extract_entities("Some text.")
    assert result == []
    mock_post.assert_not_called()


def test_extract_relationships_filters_by_confidence():
    entities = [
        {"canonical_name": "Acme", "entity_type": "ORGANIZATION", "aliases": []},
        {"canonical_name": "Bob", "entity_type": "PERSON", "aliases": []},
    ]
    payload = json.dumps([
        {"source": "Acme", "target": "Bob", "type": "EMPLOYS", "confidence": 0.9},
        {"source": "Acme", "target": "Bob", "type": "MENTIONS", "confidence": 0.4},
    ])
    with patch("rag.graph_extraction.requests.post", return_value=_mock_llm_response(payload)):
        from rag.graph_extraction import extract_relationships
        result = extract_relationships("Acme hired Bob.", entities)
    assert len(result) == 1
    assert result[0]["type"] == "EMPLOYS"


def test_extract_relationships_empty_entities_returns_empty():
    from rag.graph_extraction import extract_relationships
    with patch("rag.graph_extraction.requests.post") as mock_post:
        result = extract_relationships("Some text.", [])
    assert result == []
    mock_post.assert_not_called()


def test_extract_relationships_returns_empty_on_api_error():
    entities = [{"canonical_name": "Acme", "entity_type": "ORGANIZATION", "aliases": []}]
    with patch("rag.graph_extraction.requests.post", side_effect=Exception("fail")):
        from rag.graph_extraction import extract_relationships
        result = extract_relationships("Some text.", entities)
    assert result == []


def test_store_entities_and_edges_inserts_to_postgres():
    conn = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    entities = [{"canonical_name": "Acme", "entity_type": "ORGANIZATION", "aliases": ["Acme Inc"]}]
    from rag.graph_extraction import store_entities_and_edges
    entity_ids = store_entities_and_edges(conn, driver, "chunk-uuid", "source-uuid", entities, [])

    assert len(entity_ids) == 1
    assert conn.execute.called
    insert_call_args = conn.execute.call_args_list[0][0]
    assert "INSERT INTO entities" in insert_call_args[0]


def test_store_entities_and_edges_creates_memgraph_nodes():
    conn = MagicMock()
    session_mock = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session_mock)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    entities = [{"canonical_name": "Acme", "entity_type": "ORGANIZATION", "aliases": []}]
    from rag.graph_extraction import store_entities_and_edges
    store_entities_and_edges(conn, driver, "chunk-uuid", "source-uuid", entities, [])

    cypher_calls = [str(c) for c in session_mock.run.call_args_list]
    assert any("MERGE" in c and "Entity" in c for c in cypher_calls)
    assert any("MENTIONS" in c for c in cypher_calls)


def test_store_entities_and_edges_creates_related_to_edges():
    conn = MagicMock()
    session_mock = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session_mock)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    entities = [
        {"canonical_name": "Acme", "entity_type": "ORGANIZATION", "aliases": []},
        {"canonical_name": "Bob", "entity_type": "PERSON", "aliases": []},
    ]
    rels = [{"source": "Acme", "target": "Bob", "type": "EMPLOYS", "confidence": 0.9}]
    from rag.graph_extraction import store_entities_and_edges
    store_entities_and_edges(conn, driver, "chunk-uuid", "source-uuid", entities, rels)

    cypher_calls = [str(c) for c in session_mock.run.call_args_list]
    assert any("RELATED_TO" in c for c in cypher_calls)


def test_extract_and_store_graph_iterates_all_chunks():
    conn = MagicMock()
    driver = MagicMock()
    chunk_rows = [("chunk-1", "Text one."), ("chunk-2", "Text two.")]

    with patch("rag.graph_extraction.extract_entities", return_value=[]) as mock_ent, \
         patch("rag.graph_extraction.extract_relationships", return_value=[]) as mock_rel, \
         patch("rag.graph_extraction.store_entities_and_edges") as mock_store:
        from rag.graph_extraction import extract_and_store_graph
        extract_and_store_graph(conn, driver, "source-uuid", "job-uuid", chunk_rows)

    assert mock_ent.call_count == 2
    assert mock_rel.call_count == 2
    assert mock_store.call_count == 2
