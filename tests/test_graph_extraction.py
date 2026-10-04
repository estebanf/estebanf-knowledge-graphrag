import json
from unittest.mock import patch, MagicMock

import httpx
import pytest


def _mock_opencode(monkeypatch, handler):
    """Route the shared OpenCode client through an `httpx.MockTransport`."""
    monkeypatch.setattr(
        "rag.opencode._new_client",
        lambda timeout: httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout),
    )


def _responses_reply(payload) -> httpx.Response:
    """A gpt-* `/responses` reply whose message text is `payload` as JSON."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return httpx.Response(200, json={"output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": text}]},
    ]})


def _chat_reply(payload) -> httpx.Response:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


@pytest.fixture(autouse=True)
def _opencode_key(monkeypatch):
    monkeypatch.setattr("rag.graph_extraction.settings.OPENCODE_API_KEY", "test-key")
    monkeypatch.setattr("rag.graph_extraction.settings.MODEL_ENTITY_EXTRACTION", "gpt-6-luna")
    monkeypatch.setattr("rag.graph_extraction.settings.EXTRACTION_REASONING_EFFORT", "none")
    monkeypatch.setattr("rag.opencode.settings.OPENCODE_API_KEY", "test-key")
    monkeypatch.setattr("rag.opencode.settings.OPENCODE_FALLBACK_MODEL", "")
    monkeypatch.setattr("rag.opencode._sleep", lambda s: None)


def _extract_with_reply(monkeypatch, payload, text="Some text."):
    _mock_opencode(monkeypatch, lambda request: _responses_reply(payload))
    from rag.graph_extraction import extract_entities
    return extract_entities(text)


def test_extract_entities_returns_list(monkeypatch):
    result = _extract_with_reply(monkeypatch, {"entities": [
        {"canonical_name": "Acme Corp", "entity_type": "ORGANIZATION", "aliases": ["Acme"]}
    ]}, "Acme Corp acquired Widgets Inc last year.")
    assert len(result) == 1
    assert result[0]["canonical_name"] == "Acme Corp"
    assert result[0]["entity_type"] == "ORGANIZATION"


def test_extract_entities_accepts_bare_list(monkeypatch):
    result = _extract_with_reply(monkeypatch, [
        {"canonical_name": "Acme Corp", "entity_type": "ORGANIZATION", "aliases": []}
    ])
    assert [e["canonical_name"] for e in result] == ["Acme Corp"]


def test_extract_entities_empty_object_returns_empty(monkeypatch):
    assert _extract_with_reply(monkeypatch, {"entities": []}) == []


def test_extract_entities_normalises_entity_type_case_and_separators(monkeypatch):
    """gpt-6-luna returns lowercase types; without normalisation every entity
    would be silently dropped and the stage would 'succeed' with zero."""
    result = _extract_with_reply(monkeypatch, {"entities": [
        {"canonical_name": "Acme", "entity_type": "organization", "aliases": []},
        {"canonical_name": "ChatGPT", "entity_type": "Product", "aliases": []},
        {"canonical_name": "GDPR", "entity_type": " regulation ", "aliases": []},
    ]})
    assert [(e["canonical_name"], e["entity_type"]) for e in result] == [
        ("Acme", "ORGANIZATION"), ("ChatGPT", "PRODUCT"), ("GDPR", "REGULATION"),
    ]


def test_normalise_entity_type_maps_spaces_and_hyphens():
    from rag.graph_extraction import _normalise_entity_type
    assert _normalise_entity_type("named entity") == "NAMED_ENTITY"
    assert _normalise_entity_type("named-entity") == "NAMED_ENTITY"
    assert _normalise_entity_type("  person ") == "PERSON"


def test_extract_entities_drops_off_list_types_even_after_normalisation(monkeypatch):
    result = _extract_with_reply(monkeypatch, {"entities": [
        {"canonical_name": "ChatGPT", "entity_type": "product", "aliases": []},
        {"canonical_name": "LangChain", "entity_type": "software", "aliases": []},
        {"canonical_name": "Engineer", "entity_type": "ROLE", "aliases": []},
        {"canonical_name": "NoType"},
    ]})
    assert [e["canonical_name"] for e in result] == ["ChatGPT"]


def test_extract_entities_filters_invalid_type(monkeypatch):
    result = _extract_with_reply(monkeypatch, {"entities": [
        {"canonical_name": "ChatGPT", "entity_type": "PRODUCT", "aliases": []},
        {"canonical_name": "January 2024", "entity_type": "DATE", "aliases": []},
        {"canonical_name": "Engineer", "entity_type": "ROLE", "aliases": []},
    ]}, "ChatGPT launched in January 2024 for engineers.")
    assert len(result) == 1
    assert result[0]["canonical_name"] == "ChatGPT"


def test_extract_entities_normalises_html_entities(monkeypatch):
    result = _extract_with_reply(monkeypatch, {"entities": [
        {"canonical_name": "AT&amp;T", "entity_type": "ORGANIZATION", "aliases": ["AT&amp;T Inc"]},
    ]}, "AT&T is a telecom company.")
    assert len(result) == 1
    assert result[0]["canonical_name"] == "AT&T"
    assert "AT&T Inc" in result[0]["aliases"]


def test_extract_entities_sends_model_reasoning_effort_and_session(monkeypatch):
    from rag.opencode import SESSION_HEADER
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["session"] = request.headers.get(SESSION_HEADER)
        return _responses_reply({"entities": []})

    _mock_opencode(monkeypatch, handler)
    from rag.graph_extraction import extract_entities
    extract_entities("Some text.", session_id="session-xyz")

    assert seen["url"] == "https://opencode.ai/zen/go/v1/responses"
    assert seen["body"]["model"] == "gpt-6-luna"
    assert seen["body"]["reasoning"] == {"effort": "none"}
    assert "temperature" not in seen["body"]
    assert seen["session"] == "session-xyz"


def test_extract_entities_deepseek_override_routes_to_chat_completions(monkeypatch):
    monkeypatch.setattr("rag.graph_extraction.settings.MODEL_ENTITY_EXTRACTION", "deepseek-v4.1-flash")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return _chat_reply({"entities": [
            {"canonical_name": "Acme", "entity_type": "organization", "aliases": []}
        ]})

    _mock_opencode(monkeypatch, handler)
    from rag.graph_extraction import extract_entities
    result = extract_entities("Some text.")

    assert seen["url"].endswith("/chat/completions")
    assert seen["body"]["model"] == "deepseek-v4.1-flash"
    assert seen["body"]["reasoning_effort"] == "none"
    assert result[0]["entity_type"] == "ORGANIZATION"


def test_extract_entities_raises_on_api_error(monkeypatch):
    """A genuine LLM/HTTP failure is not swallowed (KTD6/R7): the parallel
    wrapper needs to distinguish this from "chunk has zero entities" so it
    can record the chunk into `failed_chunks`."""
    def boom(request):
        raise httpx.ConnectError("network error")

    _mock_opencode(monkeypatch, boom)
    from rag.graph_extraction import extract_entities
    with pytest.raises(httpx.ConnectError, match="network error"):
        extract_entities("Some text.")


def test_extract_entities_returns_empty_without_api_key(monkeypatch):
    monkeypatch.setattr("rag.graph_extraction.settings.OPENCODE_API_KEY", "")
    called = []
    _mock_opencode(monkeypatch, lambda request: called.append(request) or _responses_reply([]))
    from rag.graph_extraction import extract_entities
    assert extract_entities("Some text.") == []
    assert called == []


def test_extract_entities_ignores_openrouter_key(monkeypatch):
    """Entity extraction moved to OpenCode; an OpenRouter key alone no longer enables it."""
    monkeypatch.setattr("rag.graph_extraction.settings.OPENCODE_API_KEY", "")
    monkeypatch.setattr("rag.graph_extraction.settings.OPENROUTER_API_KEY", "or-key")
    from rag.graph_extraction import extract_entities
    assert extract_entities("Some text.") == []


def test_relationship_extraction_removed():
    from rag import graph_extraction
    assert not hasattr(graph_extraction, "extract_relationships")


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

    def fake_extract(content: str, session_id=None) -> list[dict]:
        name = content.split(" ")[0]
        return [{"canonical_name": name, "entity_type": "PERSON", "aliases": []}]

    with patch("rag.graph_extraction.extract_entities", side_effect=fake_extract), \
         patch("rag.graph_extraction.get_embeddings", return_value=[[0.1] * 4096] * 3), \
         patch("rag.graph_extraction.settings.OPENCODE_API_KEY", "test-key"):
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

    def fake_extract(content: str, session_id=None) -> list[dict]:
        if content.startswith("Bad"):
            raise RuntimeError("LLM boom")
        return []

    with patch("rag.graph_extraction.extract_entities", side_effect=fake_extract), \
         patch("rag.graph_extraction.settings.OPENCODE_API_KEY", "test-key"), \
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

    def fake_extract(content: str, session_id=None) -> list[dict]:
        if content.startswith("Bad"):
            raise RuntimeError("LLM boom")
        return []

    with patch("rag.graph_extraction.extract_entities", side_effect=fake_extract), \
         patch("rag.graph_extraction.settings.OPENCODE_API_KEY", "test-key"), \
         patch("rag.graph_extraction.settings.STAGE_FAILURE_RATE_THRESHOLD", 0.25):
        from rag.graph_extraction import extract_and_store_graph
        with pytest.raises(RuntimeError, match="failure rate"):
            extract_and_store_graph(conn, driver, "source-uuid", "job-uuid", chunk_rows)


def test_extract_and_store_graph_missing_api_key_short_circuits():
    """Missing-key path (KTD6b): unset OPENCODE_API_KEY short-circuits
    before fan-out with an explicit skip marker, not a 100% failure rate."""
    conn = MagicMock()
    driver = MagicMock()
    chunk_rows = [("chunk-1", "Text one."), ("chunk-2", "Text two.")]

    with patch("rag.graph_extraction.extract_entities") as mock_ent, \
         patch("rag.graph_extraction.settings.OPENCODE_API_KEY", ""):
        from rag.graph_extraction import extract_and_store_graph
        result = extract_and_store_graph(conn, driver, "source-uuid", "job-uuid", chunk_rows)

    mock_ent.assert_not_called()
    driver.session.assert_not_called()
    assert result["skipped"] is True
    assert result["failed_chunks"] == []
    assert result["chunks_processed"] == 0


def test_extract_and_store_graph_missing_key_reports_skip_reason(monkeypatch):
    monkeypatch.setattr("rag.graph_extraction.settings.OPENCODE_API_KEY", "")
    from rag.graph_extraction import extract_and_store_graph
    result = extract_and_store_graph(MagicMock(), MagicMock(), "s", "j", [("c1", "text")])
    assert result["skipped"] is True
    assert result["skip_reason"] == "OPENCODE_API_KEY not configured"
    assert result["failed_chunks"] == []


def test_extract_and_store_graph_shares_one_session_across_chunks(monkeypatch):
    """One source is one OpenCode conversation: every chunk call carries the
    same x-opencode-session, and a different source gets a different one."""
    from rag.opencode import SESSION_HEADER
    sessions = []

    def handler(request):
        sessions.append(request.headers.get(SESSION_HEADER))
        return _responses_reply({"entities": []})

    _mock_opencode(monkeypatch, handler)
    monkeypatch.setattr("rag.graph_extraction.settings.GRAPH_EXTRACTION_CONCURRENCY", 3)
    conn = MagicMock()
    driver, _session = _driver_with_session()
    chunk_rows = [(f"chunk-{i}", f"Text {i}.") for i in range(5)]

    from rag.graph_extraction import extract_and_store_graph
    extract_and_store_graph(conn, driver, "source-1", "job-1", chunk_rows)
    first = set(sessions)
    assert len(sessions) == 5 and len(first) == 1 and first != {None}

    sessions.clear()
    extract_and_store_graph(conn, driver, "source-2", "job-2", chunk_rows)
    assert len(set(sessions)) == 1 and set(sessions) != first


def test_extract_and_store_graph_http_failures_gate_on_threshold(monkeypatch):
    """End to end through the real client: HTTP 400s on every chunk are
    recorded as failed_chunks and trip the shared failure-rate gate."""
    _mock_opencode(monkeypatch, lambda request: httpx.Response(400, json={"error": "bad"}))
    monkeypatch.setattr("rag.graph_extraction.settings.STAGE_FAILURE_RATE_THRESHOLD", 0.25)
    conn = MagicMock()
    driver, _session = _driver_with_session()
    from rag.graph_extraction import extract_and_store_graph
    with pytest.raises(RuntimeError, match="failure rate"):
        extract_and_store_graph(conn, driver, "s", "j", [("c1", "a"), ("c2", "b")])
