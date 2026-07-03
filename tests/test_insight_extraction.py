from unittest.mock import MagicMock, patch


def test_extract_returns_empty_without_api_key(monkeypatch):
    monkeypatch.setattr("rag.insight_extraction.settings.OPENCODE_API_KEY", "")
    from rag.insight_extraction import extract_insights_from_chunk
    assert extract_insights_from_chunk("some text") == []


def test_extract_returns_empty_on_api_error(monkeypatch):
    monkeypatch.setattr("rag.insight_extraction.settings.OPENCODE_API_KEY", "test-key")
    with patch("rag.insight_extraction.httpx.post", side_effect=Exception("connection error")):
        from rag.insight_extraction import extract_insights_from_chunk
        assert extract_insights_from_chunk("some text") == []


def test_extract_parses_valid_response(monkeypatch):
    monkeypatch.setattr("rag.insight_extraction.settings.OPENCODE_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"insights": [{"insight": "AI reduces costs", "topics": ["AI Adoption"]}]}'}}]
    }
    mock_resp.raise_for_status = MagicMock()
    with patch("rag.insight_extraction.httpx.post", return_value=mock_resp):
        from rag.insight_extraction import extract_insights_from_chunk
        result = extract_insights_from_chunk("AI reduces operational costs significantly.")
    assert len(result) == 1
    assert result[0]["insight"] == "AI reduces costs"
    assert result[0]["topics"] == ["AI Adoption"]


def test_upsert_insight_reuses_existing():
    from rag.insight_extraction import upsert_insight
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = ("existing-uuid", 0.97)
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    insight_id, is_new = upsert_insight(conn, "some insight", [0.1] * 4096)
    assert insight_id == "existing-uuid"
    assert is_new is False
    # Only the prefilter+rerank SELECT ran — no INSERT was issued.
    assert cursor.execute.call_count == 1
    select_sql = cursor.execute.call_args_list[0].args[0]
    assert "INSERT" not in select_sql.upper()


def test_upsert_insight_creates_new_when_below_threshold():
    from rag.insight_extraction import upsert_insight
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.side_effect = [("existing-uuid", 0.80), ("new-uuid",)]
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    insight_id, is_new = upsert_insight(conn, "different insight", [0.9] * 4096)
    assert is_new is True
    assert insight_id == "new-uuid"
    insert_sql = cursor.execute.call_args_list[1].args[0]
    assert "INSERT" in insert_sql.upper()


def test_upsert_insight_ignores_null_embeddings():
    from rag.insight_extraction import upsert_insight
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.side_effect = [None, ("new-uuid",)]
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    upsert_insight(conn, "some insight", [0.1] * 4096)

    similarity_sql = cursor.execute.call_args_list[0].args[0]
    assert "embedding IS NOT NULL" in similarity_sql


def test_upsert_insight_uses_binary_prefilter_cte():
    """Mirrors dense_retrieve's SQL shape: a MATERIALIZED CTE prefilter over
    the binary-quantized HNSW index, reranked by full-precision cosine
    distance. This is the fix for the O(corpus) sequential scan."""
    from rag.insight_extraction import upsert_insight
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = ("existing-uuid", 0.97)
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    upsert_insight(conn, "some insight", [0.1] * 4096)

    select_sql = cursor.execute.call_args_list[0].args[0]
    assert "MATERIALIZED" in select_sql
    assert "binary_quantize(embedding)::bit(4096)" in select_sql
    assert "<~>" in select_sql
    assert "binary_quantize(%s::vector)::bit(4096)" in select_sql
    assert "embedding <=>" in select_sql or "i.embedding <=>" in select_sql


def test_upsert_insight_sets_hnsw_ef_search_to_candidate_count(monkeypatch):
    from rag.insight_extraction import upsert_insight
    from rag.config import settings
    monkeypatch.setattr(settings, "INSIGHT_PREFILTER_CANDIDATES", 137)
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = ("existing-uuid", 0.97)
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    upsert_insight(conn, "some insight", [0.1] * 4096)

    conn.execute.assert_called_once_with("SET hnsw.ef_search = 137")


def test_upsert_insight_empty_table_creates_new_without_error():
    """Empty insights table: prefilter CTE returns zero rows, the rerank
    join yields no candidates, and the function falls through to INSERT
    with no error raised."""
    from rag.insight_extraction import upsert_insight
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.side_effect = [None, ("new-uuid",)]
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    insight_id, is_new = upsert_insight(conn, "first insight", [0.1] * 4096)

    assert insight_id == "new-uuid"
    assert is_new is True


def test_upsert_insight_all_prefilter_candidates_below_threshold():
    """Prefilter returns candidates, but the best one after full-precision
    rerank is still below INSIGHT_DEDUP_COSINE_THRESHOLD → new insight."""
    from rag.insight_extraction import upsert_insight
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.side_effect = [("closest-uuid", 0.42), ("new-uuid",)]
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    insight_id, is_new = upsert_insight(conn, "novel insight", [0.5] * 4096)

    assert is_new is True
    assert insight_id == "new-uuid"


def test_link_chunk_insight_executes_upsert_sql():
    from rag.insight_extraction import link_chunk_insight
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    link_chunk_insight(conn, "chunk-id", "insight-id", ["AI Adoption"])
    sql_called = cursor.execute.call_args[0][0]
    assert "ON CONFLICT" in sql_called
    assert "chunk_insights" in sql_called


def test_store_insight_in_graph_merges_node_and_edge():
    from rag.insight_extraction import store_insight_in_graph
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = lambda s: session
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    store_insight_in_graph(driver, "chunk-1", "insight-1", "AI reduces costs", ["AI Adoption"])
    assert session.run.call_count == 2
    first_call_cypher = session.run.call_args_list[0][0][0]
    assert "MERGE" in first_call_cypher and "Insight" in first_call_cypher
    second_call_cypher = session.run.call_args_list[1][0][0]
    assert "CONTAINS" in second_call_cypher


def _make_cursor_mock(conn):
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return cursor


def _make_session_mock(driver):
    session = MagicMock()
    driver.session.return_value.__enter__ = lambda s: session
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return session


def test_link_related_insights_creates_mutual_edge_with_forward_similarity():
    """Two insights mutually in each other's top-K, different sources ->
    one bidirectional RELATED_TO pair written with the forward similarity."""
    from rag.insight_extraction import link_related_insights
    conn = MagicMock()
    driver = MagicMock()
    session = _make_session_mock(driver)
    cursor = _make_cursor_mock(conn)
    b_emb = [0.2] * 4096
    cursor.fetchall.side_effect = [
        [("b-id", 0.88, b_emb)],   # forward pass: A's top-K -> B
        [("a-id",)],               # reverse pass: B's top-K -> includes A, mutual
    ]

    link_related_insights(conn, driver, "source-a", [("a-id", [0.1] * 4096)])

    assert session.run.call_count == 1
    call = session.run.call_args
    cypher = call.args[0]
    assert "RELATED_TO" in cypher
    edges = call.kwargs["edges"]
    assert len(edges) == 1
    assert {edges[0]["a_id"], edges[0]["b_id"]} == {"a-id", "b-id"}
    assert edges[0]["sim"] == 0.88


def test_link_related_insights_skips_non_mutual():
    """A has B in its top-K but B does not have A in its top-K -> no edge."""
    from rag.insight_extraction import link_related_insights
    conn = MagicMock()
    driver = MagicMock()
    session = _make_session_mock(driver)
    cursor = _make_cursor_mock(conn)
    cursor.fetchall.side_effect = [
        [("b-id", 0.88, [0.2] * 4096)],   # forward pass: A's top-K -> B
        [("c-id",)],                        # reverse pass: B's top-K -> C, not A
    ]

    link_related_insights(conn, driver, "source-a", [("a-id", [0.1] * 4096)])

    session.run.assert_not_called()


def test_link_related_insights_excludes_current_source_candidates():
    """A same-source candidate is never linked, even at the highest
    similarity — the exclusion runs inside the forward-pass SQL itself."""
    from rag.insight_extraction import link_related_insights
    conn = MagicMock()
    driver = MagicMock()
    _make_session_mock(driver)
    cursor = _make_cursor_mock(conn)
    cursor.fetchall.return_value = []

    link_related_insights(conn, driver, "source-a", [("a-id", [0.1] * 4096)])

    sql = cursor.execute.call_args_list[0].args[0]
    params = cursor.execute.call_args_list[0].args[1]
    assert "NOT EXISTS" in sql
    assert "c.source_id = %s" in sql
    assert params[0] == "a-id"
    assert params[4] == "source-a"


def test_link_related_insights_writes_batch_edges_in_one_memgraph_call():
    """Three new insights each produce a mutual edge -> all edges written
    via a single Memgraph session.run call, not one per pair."""
    from rag.insight_extraction import link_related_insights
    conn = MagicMock()
    driver = MagicMock()
    session = _make_session_mock(driver)
    cursor = _make_cursor_mock(conn)
    cursor.fetchall.side_effect = [
        [("b1-id", 0.81, [0.21] * 4096)],  # forward: a1 -> b1
        [("b2-id", 0.82, [0.22] * 4096)],  # forward: a2 -> b2
        [("b3-id", 0.83, [0.23] * 4096)],  # forward: a3 -> b3
        [("a1-id",)],                        # reverse: b1 -> a1 (mutual)
        [("a2-id",)],                        # reverse: b2 -> a2 (mutual)
        [("a3-id",)],                        # reverse: b3 -> a3 (mutual)
    ]

    link_related_insights(
        conn,
        driver,
        "source-a",
        [
            ("a1-id", [0.1] * 4096),
            ("a2-id", [0.1] * 4096),
            ("a3-id", [0.1] * 4096),
        ],
    )

    assert session.run.call_count == 1
    edges = session.run.call_args.kwargs["edges"]
    assert len(edges) == 3
    pairs = {frozenset((e["a_id"], e["b_id"])) for e in edges}
    assert pairs == {
        frozenset(("a1-id", "b1-id")),
        frozenset(("a2-id", "b2-id")),
        frozenset(("a3-id", "b3-id")),
    }


def test_link_related_insights_no_candidates_makes_no_memgraph_call():
    """A batch of new insights with zero qualifying candidates -> no
    Memgraph call is made, and nothing raises."""
    from rag.insight_extraction import link_related_insights
    conn = MagicMock()
    driver = MagicMock()
    cursor = _make_cursor_mock(conn)
    cursor.fetchall.return_value = []

    link_related_insights(conn, driver, "source-a", [("a-id", [0.1] * 4096)])

    driver.session.assert_not_called()


def test_link_related_insights_empty_batch_is_a_noop():
    from rag.insight_extraction import link_related_insights
    conn = MagicMock()
    driver = MagicMock()

    link_related_insights(conn, driver, "source-a", [])

    conn.cursor.assert_not_called()
    driver.session.assert_not_called()


def test_link_related_insights_accepts_database_vector_strings():
    from rag.insight_extraction import link_related_insights
    conn = MagicMock()
    driver = MagicMock()
    session = _make_session_mock(driver)
    cursor = _make_cursor_mock(conn)
    cursor.fetchall.side_effect = [
        [("b-id", 0.88, "[0.2,0.2]")],
        [("a-id",)],
    ]

    link_related_insights(conn, driver, "source-a", [("a-id", [0.1] * 2)])

    reverse_call_params = cursor.execute.call_args_list[1].args[1]
    assert reverse_call_params[1] == "[0.2,0.2]"
    assert session.run.call_count == 1


def test_extract_chunk_insights_parallel_preserves_chunk_order(monkeypatch):
    from rag.insight_extraction import _extract_chunk_insights_parallel

    monkeypatch.setattr("rag.insight_extraction.settings.INSIGHT_EXTRACTION_CONCURRENCY", 3)

    def fake_extract(content):
        return [{"insight": f"insight {content}", "topics": [content.upper()]}]

    monkeypatch.setattr("rag.insight_extraction.extract_insights_from_chunk", fake_extract)

    result = _extract_chunk_insights_parallel(
        [
            ("chunk-1", "alpha"),
            ("chunk-2", "beta"),
            ("chunk-3", "gamma"),
        ]
    )

    assert result == [
        ("chunk-1", "alpha", [{"insight": "insight alpha", "topics": ["ALPHA"]}]),
        ("chunk-2", "beta", [{"insight": "insight beta", "topics": ["BETA"]}]),
        ("chunk-3", "gamma", [{"insight": "insight gamma", "topics": ["GAMMA"]}]),
    ]


def test_extract_chunk_insights_parallel_reports_progress(monkeypatch):
    from rag.insight_extraction import _extract_chunk_insights_parallel

    monkeypatch.setattr("rag.insight_extraction.settings.INSIGHT_EXTRACTION_CONCURRENCY", 2)
    monkeypatch.setattr(
        "rag.insight_extraction.extract_insights_from_chunk",
        lambda content: [{"insight": f"insight {content}", "topics": []}],
    )
    events = []

    _extract_chunk_insights_parallel(
        [("chunk-1", "alpha"), ("chunk-2", "beta")],
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    assert events[0] == ("extract_start", {"total": 2, "concurrency": 2})
    assert [event for event, _payload in events].count("extract_chunk") == 2
    assert events[-1] == ("extract_done", {"total": 2})


def test_extract_and_store_insights_returns_counts(monkeypatch):
    monkeypatch.setattr("rag.insight_extraction.extract_insights_from_chunk",
                        lambda content: [{"insight": "insight A", "topics": ["AI Adoption"]}])
    monkeypatch.setattr("rag.insight_extraction.upsert_insight",
                        lambda conn, content, emb: ("new-uuid", True))
    monkeypatch.setattr("rag.insight_extraction.link_chunk_insight", lambda *a, **k: None)
    monkeypatch.setattr("rag.insight_extraction.store_insight_in_graph", lambda *a, **k: None)
    monkeypatch.setattr("rag.insight_extraction.link_related_insights", lambda *a, **k: None)
    monkeypatch.setattr("rag.insight_extraction.get_embeddings", lambda texts: [[0.1]*4096])

    from rag.insight_extraction import extract_and_store_insights
    conn = MagicMock()
    driver = MagicMock()
    result = extract_and_store_insights(
        conn, driver, "src-id", [("chunk-1", "some content")]
    )
    assert result["chunks_processed"] == 1
    assert result["insights_extracted"] == 1
    assert result["insights_reused"] == 0


def test_extract_and_store_insights_stores_parallel_results_serially(monkeypatch):
    events = []

    monkeypatch.setattr(
        "rag.insight_extraction._extract_chunk_insights_parallel",
        lambda rows, progress_callback=None: [
            ("chunk-1", "content 1", [{"insight": "insight A", "topics": ["AI Adoption"]}]),
            ("chunk-2", "content 2", [{"insight": "insight B", "topics": ["Business Outcomes"]}]),
        ],
    )
    monkeypatch.setattr("rag.insight_extraction.get_embeddings", lambda texts: [[0.1] * 4096])
    monkeypatch.setattr(
        "rag.insight_extraction.upsert_insight",
        lambda conn, content, emb: (events.append(("upsert", content)) or (f"{content}-id", True)),
    )
    monkeypatch.setattr(
        "rag.insight_extraction.link_chunk_insight",
        lambda conn, chunk_id, insight_id, topics: events.append(("link_chunk", chunk_id, insight_id)),
    )
    monkeypatch.setattr(
        "rag.insight_extraction.store_insight_in_graph",
        lambda driver, chunk_id, insight_id, content, topics: events.append(("graph", chunk_id, insight_id)),
    )
    monkeypatch.setattr(
        "rag.insight_extraction.link_related_insights",
        lambda conn, driver, source_id, insight_id, emb: events.append(("related", source_id, insight_id)),
    )

    from rag.insight_extraction import extract_and_store_insights
    conn = MagicMock()
    driver = MagicMock()

    result = extract_and_store_insights(
        conn, driver, "source-1", [("chunk-1", "content 1"), ("chunk-2", "content 2")]
    )

    assert result["chunks_processed"] == 2
    assert events == [
        ("upsert", "insight A"),
        ("link_chunk", "chunk-1", "insight A-id"),
        ("graph", "chunk-1", "insight A-id"),
        ("related", "source-1", "insight A-id"),
        ("upsert", "insight B"),
        ("link_chunk", "chunk-2", "insight B-id"),
        ("graph", "chunk-2", "insight B-id"),
        ("related", "source-1", "insight B-id"),
    ]


def test_extract_and_store_insights_reports_storage_progress(monkeypatch):
    monkeypatch.setattr(
        "rag.insight_extraction._extract_chunk_insights_parallel",
        lambda rows, progress_callback=None: [
            ("chunk-1", "content 1", [{"insight": "insight A", "topics": []}]),
            ("chunk-2", "content 2", []),
        ],
    )
    monkeypatch.setattr("rag.insight_extraction.get_embeddings", lambda texts: [[0.1] * 4096])
    monkeypatch.setattr("rag.insight_extraction.upsert_insight", lambda *a, **k: ("insight-1", True))
    monkeypatch.setattr("rag.insight_extraction.link_chunk_insight", lambda *a, **k: None)
    monkeypatch.setattr("rag.insight_extraction.store_insight_in_graph", lambda *a, **k: None)
    monkeypatch.setattr("rag.insight_extraction.link_related_insights", lambda *a, **k: None)
    events = []

    from rag.insight_extraction import extract_and_store_insights
    result = extract_and_store_insights(
        MagicMock(),
        MagicMock(),
        "source-1",
        [("chunk-1", "content 1"), ("chunk-2", "content 2")],
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    assert result["chunks_processed"] == 2
    assert events == [
        ("store_start", {"total": 2}),
        ("store_chunk", {"position": 1, "total": 2, "chunk_id": "chunk-1", "insights": 1}),
        ("store_chunk", {"position": 2, "total": 2, "chunk_id": "chunk-2", "insights": 0}),
        ("store_done", {"total": 2}),
    ]


def test_extract_and_store_insights_skips_blank_insight(monkeypatch):
    monkeypatch.setattr(
        "rag.insight_extraction.extract_insights_from_chunk",
        lambda content: [
            {"insight": "", "topics": ["AI Adoption"]},
            {"topics": ["AI Adoption"]},
        ],
    )
    mock_embeddings = MagicMock()
    monkeypatch.setattr("rag.insight_extraction.get_embeddings", mock_embeddings)

    from rag.insight_extraction import extract_and_store_insights
    conn = MagicMock()
    driver = MagicMock()

    result = extract_and_store_insights(
        conn, driver, "src-id", [("chunk-1", "some content")]
    )

    assert result["chunks_processed"] == 1
    assert result["insights_extracted"] == 0
    assert result["insights_reused"] == 0
    mock_embeddings.assert_not_called()
