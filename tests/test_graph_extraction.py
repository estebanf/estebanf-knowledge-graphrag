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


def test_extract_entities_raises_on_api_error():
    """A genuine LLM/HTTP failure is no longer swallowed (KTD6/R7): the
    parallel wrapper needs to distinguish this from "chunk has zero
    entities" so it can record the chunk into `failed_chunks`."""
    with patch("rag.graph_extraction.requests.post", side_effect=Exception("network error")):
        from rag.graph_extraction import extract_entities
        with pytest.raises(Exception, match="network error"):
            extract_entities("Some text.")


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


def test_extract_entities_filters_invalid_type():
    payload = json.dumps([
        {"canonical_name": "ChatGPT", "entity_type": "PRODUCT", "aliases": []},
        {"canonical_name": "January 2024", "entity_type": "DATE", "aliases": []},
        {"canonical_name": "Engineer", "entity_type": "ROLE", "aliases": []},
    ])
    with patch("rag.graph_extraction.requests.post", return_value=_mock_llm_response(payload)):
        from rag.graph_extraction import extract_entities
        result = extract_entities("ChatGPT launched in January 2024 for engineers.")
    assert len(result) == 1
    assert result[0]["canonical_name"] == "ChatGPT"
    assert all(e["entity_type"] != "DATE" for e in result)
    assert all(e["entity_type"] != "ROLE" for e in result)


def test_extract_entities_normalises_html_entities():
    payload = json.dumps([
        {"canonical_name": "AT&amp;T", "entity_type": "ORGANIZATION", "aliases": ["AT&amp;T Inc"]},
    ])
    with patch("rag.graph_extraction.requests.post", return_value=_mock_llm_response(payload)):
        from rag.graph_extraction import extract_entities
        result = extract_entities("AT&T is a telecom company.")
    assert len(result) == 1
    assert result[0]["canonical_name"] == "AT&T"
    assert "AT&T Inc" in result[0]["aliases"]


def _driver_with_session(session_mock=None):
    session_mock = session_mock or MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session_mock)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session_mock


def test_store_entities_and_edges_inserts_to_postgres():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None  # no existing entity
    driver, _session = _driver_with_session()

    entities = [{"canonical_name": "Acme", "entity_type": "ORGANIZATION", "aliases": ["Acme Inc"]}]
    with patch("rag.graph_extraction.get_embeddings", return_value=[[0.1] * 4096]):
        from rag.graph_extraction import store_entities_and_edges
        entity_ids = store_entities_and_edges(
            conn, driver, "source-uuid", [("chunk-uuid", entities)]
        )

    assert len(entity_ids) == 1
    insert_calls = [c for c in conn.execute.call_args_list if "INSERT INTO entities" in str(c)]
    assert len(insert_calls) == 1


def test_store_entities_and_edges_stores_embedding():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    driver, _session = _driver_with_session()

    entities = [{"canonical_name": "Acme", "entity_type": "ORGANIZATION", "aliases": []}]
    fake_vec = [0.1] * 4096

    with patch("rag.graph_extraction.get_embeddings", return_value=[fake_vec]) as mock_embed:
        from rag.graph_extraction import store_entities_and_edges
        store_entities_and_edges(conn, driver, "source-uuid", [("chunk-uuid", entities)])

    mock_embed.assert_called_once_with(["Acme"])
    insert_calls = [c for c in conn.execute.call_args_list if "INSERT INTO entities" in str(c)]
    assert len(insert_calls) == 1
    insert_sql, insert_params = insert_calls[0][0]
    assert "embedding" in insert_sql
    assert f"[{','.join(str(v) for v in fake_vec)}]" in insert_params


def test_store_entities_and_edges_reuses_existing_entity():
    conn = MagicMock()
    existing_id = "existing-uuid-1234"
    conn.execute.return_value.fetchone.return_value = (existing_id,)
    driver, _session = _driver_with_session()

    entities = [{"canonical_name": "Acme", "entity_type": "ORGANIZATION", "aliases": []}]
    with patch("rag.graph_extraction.get_embeddings", return_value=[[0.1] * 4096]):
        from rag.graph_extraction import store_entities_and_edges
        entity_ids = store_entities_and_edges(
            conn, driver, "source-uuid", [("chunk-uuid", entities)]
        )

    assert entity_ids == [existing_id]
    insert_calls = [c for c in conn.execute.call_args_list if "INSERT INTO entities" in str(c)]
    assert len(insert_calls) == 0
    select_calls = [c for c in conn.execute.call_args_list if "SELECT id" in str(c)]
    assert len(select_calls) == 1
    assert select_calls[0][0][1] == ("Acme",)


def test_store_entities_and_edges_batches_memgraph_writes():
    """Batching (U4/KTD4): entity nodes + MENTIONS edges for a multi-entity,
    multi-chunk batch are written via exactly two `session.run` UNWIND
    calls total, not one call pair per chunk."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    driver, session_mock = _driver_with_session()

    chunk_entities = [
        ("chunk-1", [
            {"canonical_name": "Acme", "entity_type": "ORGANIZATION", "aliases": []},
            {"canonical_name": "Bob", "entity_type": "PERSON", "aliases": []},
        ]),
        ("chunk-2", [
            {"canonical_name": "Carol", "entity_type": "PERSON", "aliases": []},
        ]),
    ]
    with patch("rag.graph_extraction.get_embeddings", return_value=[[0.1] * 4096] * 3):
        from rag.graph_extraction import store_entities_and_edges
        store_entities_and_edges(conn, driver, "source-uuid", chunk_entities)

    assert session_mock.run.call_count == 2
    cypher_calls = [str(c) for c in session_mock.run.call_args_list]
    assert any("UNWIND" in c and "Entity" in c for c in cypher_calls)
    assert any("UNWIND" in c and "MENTIONS" in c for c in cypher_calls)
    assert not any("RELATED_TO" in c for c in cypher_calls)


def test_store_entities_and_edges_dedups_same_name_across_chunks():
    """Concurrency-race note: two same-name entities from different chunks
    in one batch resolve to a single Postgres row because the single serial
    storage pass performs the SELECT-first dedup in-process order, not two
    independent concurrent lookups."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    driver, session_mock = _driver_with_session()

    chunk_entities = [
        ("chunk-1", [{"canonical_name": "Acme", "entity_type": "ORGANIZATION", "aliases": []}]),
        ("chunk-2", [{"canonical_name": "Acme", "entity_type": "ORGANIZATION", "aliases": []}]),
    ]
    with patch("rag.graph_extraction.get_embeddings", return_value=[[0.1] * 4096] * 2):
        from rag.graph_extraction import store_entities_and_edges
        entity_ids = store_entities_and_edges(conn, driver, "source-uuid", chunk_entities)

    assert len(set(entity_ids)) == 1
    insert_calls = [c for c in conn.execute.call_args_list if "INSERT INTO entities" in str(c)]
    assert len(insert_calls) == 1

    # Both chunks still get their own MENTIONS edge for the one shared node.
    edges_call = next(c for c in session_mock.run.call_args_list if "MENTIONS" in str(c))
    edges = edges_call.kwargs["edges"]
    assert {e["chunk_id"] for e in edges} == {"chunk-1", "chunk-2"}


def test_store_entities_and_edges_empty_input_skips_driver():
    conn = MagicMock()
    driver = MagicMock()
    from rag.graph_extraction import store_entities_and_edges
    entity_ids = store_entities_and_edges(conn, driver, "source-uuid", [])
    assert entity_ids == []
    driver.session.assert_not_called()


def test_extract_and_store_graph_skips_relationship_extraction():
    """No-scope-change guard (U4): extract_relationships is never called and
    no RELATED_TO edges are written by the entity-only pipeline."""
    conn = MagicMock()
    driver, session_mock = _driver_with_session()
    chunk_rows = [("chunk-1", "Text one.")]

    with patch("rag.graph_extraction.extract_entities", return_value=[]) as mock_ent, \
         patch("rag.graph_extraction.extract_relationships") as mock_rel, \
         patch("rag.graph_extraction.settings.OPENROUTER_API_KEY", "test-key"):
        from rag.graph_extraction import extract_and_store_graph
        extract_and_store_graph(conn, driver, "source-uuid", "job-uuid", chunk_rows)

    mock_ent.assert_called_once_with("Text one.")
    mock_rel.assert_not_called()
    assert not any("RELATED_TO" in str(c) for c in session_mock.run.call_args_list)


def test_extract_and_store_graph_happy_path_attributes_by_chunk():
    """Happy path: N chunks produce entities from all chunks, attributed to
    the right chunk despite concurrent completion order."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    driver, session_mock = _driver_with_session()

    chunk_rows = [
        ("chunk-1", "Acme text."),
        ("chunk-2", "Bob text."),
        ("chunk-3", "Carol text."),
    ]

    def fake_extract(content: str) -> list[dict]:
        name = content.split(" ")[0]
        return [{"canonical_name": name, "entity_type": "PERSON", "aliases": []}]

    with patch("rag.graph_extraction.extract_entities", side_effect=fake_extract), \
         patch("rag.graph_extraction.get_embeddings", return_value=[[0.1] * 4096] * 3), \
         patch("rag.graph_extraction.settings.OPENROUTER_API_KEY", "test-key"):
        from rag.graph_extraction import extract_and_store_graph
        result = extract_and_store_graph(conn, driver, "source-uuid", "job-uuid", chunk_rows)

    assert result["chunks_processed"] == 3
    assert result["entities_stored"] == 3
    assert result["failed_chunks"] == []

    edges_call = next(c for c in session_mock.run.call_args_list if "MENTIONS" in str(c))
    edges = edges_call.kwargs["edges"]
    assert {e["chunk_id"] for e in edges} == {"chunk-1", "chunk-2", "chunk-3"}


def test_extract_and_store_graph_records_failed_chunks_below_threshold():
    """Failure: one chunk's entity LLM call fails -> recorded in
    failed_chunks, other chunks unaffected; stage does not raise when under
    the shared STAGE_FAILURE_RATE_THRESHOLD."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    driver, _session = _driver_with_session()

    chunk_rows = [
        ("chunk-1", "Good text."),
        ("chunk-2", "Bad text."),
        ("chunk-3", "Good text."),
        ("chunk-4", "Good text."),
        ("chunk-5", "Good text."),
    ]

    def fake_extract(content: str) -> list[dict]:
        if content.startswith("Bad"):
            raise RuntimeError("LLM boom")
        return []

    with patch("rag.graph_extraction.extract_entities", side_effect=fake_extract), \
         patch("rag.graph_extraction.settings.OPENROUTER_API_KEY", "test-key"), \
         patch("rag.graph_extraction.settings.STAGE_FAILURE_RATE_THRESHOLD", 0.25):
        from rag.graph_extraction import extract_and_store_graph
        result = extract_and_store_graph(conn, driver, "source-uuid", "job-uuid", chunk_rows)

    assert result["failed_chunks"] == ["chunk-2"]
    assert result["chunks_processed"] == 4


def test_extract_and_store_graph_raises_over_failure_threshold():
    conn = MagicMock()
    driver, _session = _driver_with_session()

    chunk_rows = [
        ("chunk-1", "Bad text."),
        ("chunk-2", "Bad text."),
        ("chunk-3", "Good text."),
    ]

    def fake_extract(content: str) -> list[dict]:
        if content.startswith("Bad"):
            raise RuntimeError("LLM boom")
        return []

    with patch("rag.graph_extraction.extract_entities", side_effect=fake_extract), \
         patch("rag.graph_extraction.settings.OPENROUTER_API_KEY", "test-key"), \
         patch("rag.graph_extraction.settings.STAGE_FAILURE_RATE_THRESHOLD", 0.25):
        from rag.graph_extraction import extract_and_store_graph
        with pytest.raises(RuntimeError, match="failure rate"):
            extract_and_store_graph(conn, driver, "source-uuid", "job-uuid", chunk_rows)


def test_extract_and_store_graph_missing_api_key_short_circuits():
    """Missing-key path (KTD6b): unset OPENROUTER_API_KEY short-circuits
    before fan-out with an explicit skip marker, not a 100% failure rate."""
    conn = MagicMock()
    driver = MagicMock()
    chunk_rows = [("chunk-1", "Text one."), ("chunk-2", "Text two.")]

    with patch("rag.graph_extraction.extract_entities") as mock_ent, \
         patch("rag.graph_extraction.settings.OPENROUTER_API_KEY", ""):
        from rag.graph_extraction import extract_and_store_graph
        result = extract_and_store_graph(conn, driver, "source-uuid", "job-uuid", chunk_rows)

    mock_ent.assert_not_called()
    driver.session.assert_not_called()
    assert result["skipped"] is True
    assert result["failed_chunks"] == []
    assert result["chunks_processed"] == 0


def test_extract_and_store_graph_never_calls_extract_relationships_module_wide():
    """Grep-style guard: extract_relationships must not appear as a call
    target anywhere reachable from extract_and_store_graph, even indirectly
    through store_entities_and_edges."""
    import inspect
    from rag import graph_extraction

    source = inspect.getsource(graph_extraction.extract_and_store_graph)
    assert "extract_relationships(" not in source
    source = inspect.getsource(graph_extraction.store_entities_and_edges)
    assert "extract_relationships(" not in source
