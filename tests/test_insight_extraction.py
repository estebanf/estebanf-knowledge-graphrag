from unittest.mock import MagicMock, patch


def _one_hot(index: int, dim: int = 4096) -> list[float]:
    """An orthogonal-ish embedding for test fixtures: distinct indices
    produce vectors with cosine similarity 0, unlike uniform vectors (e.g.
    [0.1] * dim), which are always parallel (cosine similarity 1.0) to any
    other uniform vector regardless of magnitude and would be wrongly
    treated as within-batch duplicates by `_cosine_similarity`."""
    vec = [0.0] * dim
    vec[index % dim] = 1.0
    return vec


def test_extract_returns_empty_without_api_key(monkeypatch):
    monkeypatch.setattr("rag.insight_extraction.settings.OPENCODE_API_KEY", "")
    from rag.insight_extraction import extract_insights_from_chunk
    assert extract_insights_from_chunk("some text") == []


def test_extract_raises_on_api_error(monkeypatch):
    """Contract change (R7/KTD6): a genuine LLM-call failure is no longer
    swallowed into an empty list — it raises, so the parallel wrapper can
    distinguish it from a chunk that legitimately extracted zero insights."""
    monkeypatch.setattr("rag.insight_extraction.settings.OPENCODE_API_KEY", "test-key")
    with patch("rag.insight_extraction.httpx.post", side_effect=Exception("connection error")):
        from rag.insight_extraction import extract_insights_from_chunk
        try:
            extract_insights_from_chunk("some text")
            assert False, "expected extract_insights_from_chunk to raise"
        except Exception as exc:
            assert "connection error" in str(exc)


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


def test_store_insights_in_graph_batch_writes_two_unwind_calls():
    """Replacement for the old per-insight store_insight_in_graph: a whole
    source's chunk-insight pairs are written via exactly two UNWIND batches
    (Insight node MERGEs, CONTAINS edge MERGEs), not one MERGE pair per pair."""
    from rag.insight_extraction import store_insights_in_graph_batch
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = lambda s: session
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    store_insights_in_graph_batch(
        driver,
        [
            ("chunk-1", "insight-1", "AI reduces costs", ["AI Adoption"]),
            ("chunk-2", "insight-1", "AI reduces costs", ["AI Adoption"]),
            ("chunk-2", "insight-2", "AI improves speed", ["Performance"]),
        ],
    )

    assert session.run.call_count == 2
    node_call = session.run.call_args_list[0]
    assert "UNWIND" in node_call.args[0] and "Insight" in node_call.args[0]
    nodes = node_call.kwargs["nodes"]
    assert len(nodes) == 2  # deduped by insight id despite insight-1 appearing twice

    edge_call = session.run.call_args_list[1]
    assert "CONTAINS" in edge_call.args[0]
    edges = edge_call.kwargs["edges"]
    assert len(edges) == 3


def test_store_insights_in_graph_batch_dedupes_repeated_edge_pairs():
    """The same (chunk_id, insight_id) pair appearing twice (e.g. two raw
    insights from one chunk collapsing onto the same survivor) yields one
    edge write, not two."""
    from rag.insight_extraction import store_insights_in_graph_batch
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = lambda s: session
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    store_insights_in_graph_batch(
        driver,
        [
            ("chunk-1", "insight-1", "AI reduces costs", ["AI Adoption"]),
            ("chunk-1", "insight-1", "AI reduces costs", ["AI Adoption"]),
        ],
    )

    edges = session.run.call_args_list[1].kwargs["edges"]
    assert len(edges) == 1


def test_store_insights_in_graph_batch_noop_without_driver_or_pairs():
    from rag.insight_extraction import store_insights_in_graph_batch
    store_insights_in_graph_batch(None, [("chunk-1", "insight-1", "content", [])])
    driver = MagicMock()
    store_insights_in_graph_batch(driver, [])
    driver.session.assert_not_called()


def test_link_chunk_insights_batch_uses_executemany():
    from rag.insight_extraction import link_chunk_insights_batch
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    link_chunk_insights_batch(
        conn,
        [
            ("chunk-1", "insight-1", ["AI Adoption"]),
            ("chunk-2", "insight-1", ["AI Adoption"]),
        ],
    )

    cursor.executemany.assert_called_once()
    sql = cursor.executemany.call_args[0][0]
    params = cursor.executemany.call_args[0][1]
    assert "ON CONFLICT" in sql
    assert len(params) == 2


def test_link_chunk_insights_batch_empty_is_noop():
    from rag.insight_extraction import link_chunk_insights_batch
    conn = MagicMock()
    link_chunk_insights_batch(conn, [])
    conn.cursor.assert_not_called()


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

    result, failed_chunks = _extract_chunk_insights_parallel(
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
    assert failed_chunks == []


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


def test_extract_chunk_insights_parallel_records_failed_chunks(monkeypatch):
    """A chunk whose extraction raises is recorded in failed_chunk_ids and
    does not abort the rest of the batch (R7/KTD6) — distinct from a chunk
    that succeeds with zero insights."""
    from rag.insight_extraction import _extract_chunk_insights_parallel

    monkeypatch.setattr("rag.insight_extraction.settings.INSIGHT_EXTRACTION_CONCURRENCY", 2)

    def fake_extract(content):
        if content == "beta":
            raise RuntimeError("llm call failed")
        return [{"insight": f"insight {content}", "topics": []}]

    monkeypatch.setattr("rag.insight_extraction.extract_insights_from_chunk", fake_extract)
    events = []

    result, failed_chunks = _extract_chunk_insights_parallel(
        [("chunk-1", "alpha"), ("chunk-2", "beta"), ("chunk-3", "gamma")],
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    assert failed_chunks == ["chunk-2"]
    assert [row[0] for row in result] == ["chunk-1", "chunk-3"]
    error_events = [payload for event, payload in events if event == "extract_error"]
    assert len(error_events) == 1
    assert error_events[0]["chunk_id"] == "chunk-2"
    assert error_events[0]["error"] == "llm call failed"
    assert error_events[0]["total"] == 3


def _patch_happy_path(monkeypatch, events=None):
    """Common monkeypatches for extract_and_store_insights tests: a
    configured API key plus stubbed collaborators."""
    if events is None:
        events = []
    monkeypatch.setattr("rag.insight_extraction.settings.OPENCODE_API_KEY", "test-key")
    monkeypatch.setattr(
        "rag.insight_extraction.upsert_insight",
        lambda conn, content, emb: (events.append(("upsert", content)) or (f"{content}-id", True)),
    )
    monkeypatch.setattr(
        "rag.insight_extraction.link_chunk_insights_batch",
        lambda conn, pairs: events.append(("link_chunk_batch", pairs)),
    )
    monkeypatch.setattr(
        "rag.insight_extraction.store_insights_in_graph_batch",
        lambda driver, pairs: events.append(("graph_batch", pairs)),
    )
    monkeypatch.setattr(
        "rag.insight_extraction.link_related_insights",
        lambda conn, driver, source_id, new_insights: (
            events.append(("related", source_id, new_insights)) or len(new_insights)
        ),
    )
    return events


def test_extract_and_store_insights_returns_counts(monkeypatch):
    events = _patch_happy_path(monkeypatch)
    monkeypatch.setattr("rag.insight_extraction.extract_insights_from_chunk",
                        lambda content: [{"insight": "insight A", "topics": ["AI Adoption"]}])
    monkeypatch.setattr("rag.insight_extraction.get_embeddings", lambda texts: [[0.1] * 4096])

    from rag.insight_extraction import extract_and_store_insights
    conn = MagicMock()
    driver = MagicMock()
    result = extract_and_store_insights(
        conn, driver, "src-id", [("chunk-1", "some content")]
    )
    assert result["chunks_processed"] == 1
    assert result["insights_extracted"] == 1
    assert result["insights_reused"] == 0
    assert result["failed_chunks"] == []
    assert result["related_edges"] == 1


def test_extract_and_store_insights_batches_embeddings_across_chunks(monkeypatch):
    """Happy path: one get_embeddings call covers every insight text from
    every chunk in the source, not one call per chunk."""
    events = _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        "rag.insight_extraction._extract_chunk_insights_parallel",
        lambda rows, progress_callback=None: (
            [
                ("chunk-1", "content 1", [{"insight": "insight A", "topics": ["AI Adoption"]}]),
                ("chunk-2", "content 2", [{"insight": "insight B", "topics": ["Business Outcomes"]}]),
            ],
            [],
        ),
    )
    mock_embeddings = MagicMock(return_value=[_one_hot(0), _one_hot(1)])
    monkeypatch.setattr("rag.insight_extraction.get_embeddings", mock_embeddings)

    from rag.insight_extraction import extract_and_store_insights
    conn = MagicMock()
    driver = MagicMock()

    result = extract_and_store_insights(
        conn, driver, "source-1", [("chunk-1", "content 1"), ("chunk-2", "content 2")]
    )

    assert result["chunks_processed"] == 2
    mock_embeddings.assert_called_once_with(["insight A", "insight B"])
    assert [event for event, *_rest in events].count("upsert") == 2
    # link/graph/related happen once each for the whole batch, not per chunk.
    assert [event for event, *_rest in events].count("link_chunk_batch") == 1
    assert [event for event, *_rest in events].count("graph_batch") == 1
    assert [event for event, *_rest in events].count("related") == 1


def test_extract_and_store_insights_within_batch_duplicate_pair(monkeypatch):
    """AE2: two near-identical insights from two different chunks of the
    same source -> one upsert_insight call (one insights row) and both
    chunks link to the survivor via chunk_insights."""
    events = _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        "rag.insight_extraction._extract_chunk_insights_parallel",
        lambda rows, progress_callback=None: (
            [
                ("chunk-1", "content 1", [{"insight": "AI reduces costs", "topics": []}]),
                ("chunk-2", "content 2", [{"insight": "AI reduces expenses", "topics": []}]),
            ],
            [],
        ),
    )
    # Identical embeddings -> cosine similarity 1.0, above the dedup threshold.
    monkeypatch.setattr(
        "rag.insight_extraction.get_embeddings",
        lambda texts: [[0.5] * 4096, [0.5] * 4096],
    )

    from rag.insight_extraction import extract_and_store_insights
    conn = MagicMock()
    driver = MagicMock()

    result = extract_and_store_insights(
        conn, driver, "source-1", [("chunk-1", "content 1"), ("chunk-2", "content 2")]
    )

    upsert_calls = [payload[0] for event, *payload in events if event == "upsert"]
    assert upsert_calls == ["AI reduces costs"]  # only the first occurrence is upserted
    assert result["insights_extracted"] == 1
    assert result["insights_reused"] == 0

    link_batch_pairs = next(pairs for event, pairs in events if event == "link_chunk_batch")
    assert len(link_batch_pairs) == 2
    chunk_ids = {chunk_id for chunk_id, _insight_id, _topics in link_batch_pairs}
    insight_ids = {insight_id for _chunk_id, insight_id, _topics in link_batch_pairs}
    assert chunk_ids == {"chunk-1", "chunk-2"}
    assert len(insight_ids) == 1  # both chunks resolved onto the same survivor id


def test_extract_and_store_insights_failure_gate_allows_below_threshold(monkeypatch):
    """AE3: 2 of 12 chunks fail extraction with threshold 0.25 -> stage
    returns normally, failed_chunks lists both ids, other chunks stored."""
    monkeypatch.setattr("rag.insight_extraction.settings.OPENCODE_API_KEY", "test-key")
    monkeypatch.setattr("rag.insight_extraction.settings.STAGE_FAILURE_RATE_THRESHOLD", 0.25)
    chunk_rows = [(f"chunk-{i}", f"content {i}") for i in range(12)]
    succeeded = [
        (chunk_id, content, [{"insight": f"insight-{chunk_id}", "topics": []}])
        for chunk_id, content in chunk_rows[2:]
    ]
    monkeypatch.setattr(
        "rag.insight_extraction._extract_chunk_insights_parallel",
        lambda rows, progress_callback=None: (succeeded, ["chunk-0", "chunk-1"]),
    )
    monkeypatch.setattr(
        "rag.insight_extraction.get_embeddings",
        lambda texts: [_one_hot(i) for i in range(len(texts))],
    )
    monkeypatch.setattr("rag.insight_extraction.upsert_insight", lambda conn, content, emb: (f"{content}-id", True))
    monkeypatch.setattr("rag.insight_extraction.link_chunk_insights_batch", lambda *a, **k: None)
    monkeypatch.setattr("rag.insight_extraction.store_insights_in_graph_batch", lambda *a, **k: None)
    monkeypatch.setattr("rag.insight_extraction.link_related_insights", lambda *a, **k: 0)

    from rag.insight_extraction import extract_and_store_insights
    conn = MagicMock()
    driver = MagicMock()

    result = extract_and_store_insights(conn, driver, "source-1", chunk_rows)

    assert result["failed_chunks"] == ["chunk-0", "chunk-1"]
    assert result["chunks_processed"] == 10
    assert result["insights_extracted"] == 10


def test_extract_and_store_insights_failure_gate_raises_above_threshold(monkeypatch):
    """4 of 12 chunks fail with threshold 0.25 -> the stage raises so the
    caller's _fail_stage path can record it."""
    monkeypatch.setattr("rag.insight_extraction.settings.OPENCODE_API_KEY", "test-key")
    monkeypatch.setattr("rag.insight_extraction.settings.STAGE_FAILURE_RATE_THRESHOLD", 0.25)
    chunk_rows = [(f"chunk-{i}", f"content {i}") for i in range(12)]
    failed = [f"chunk-{i}" for i in range(4)]
    succeeded = [
        (chunk_id, content, [])
        for chunk_id, content in chunk_rows[4:]
    ]
    monkeypatch.setattr(
        "rag.insight_extraction._extract_chunk_insights_parallel",
        lambda rows, progress_callback=None: (succeeded, failed),
    )

    from rag.insight_extraction import extract_and_store_insights
    conn = MagicMock()
    driver = MagicMock()

    try:
        extract_and_store_insights(conn, driver, "source-1", chunk_rows)
        assert False, "expected extract_and_store_insights to raise"
    except RuntimeError as exc:
        assert "4/12" in str(exc)


def test_extract_and_store_insights_reuse_path_no_linking_contribution(monkeypatch):
    """Reuse path: an insight matching an existing corpus row increments
    insights_reused, and contributes nothing to link_related_insights'
    new_insights argument (linking only runs for new insights)."""
    events = _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        "rag.insight_extraction._extract_chunk_insights_parallel",
        lambda rows, progress_callback=None: (
            [("chunk-1", "content 1", [{"insight": "existing insight", "topics": []}])],
            [],
        ),
    )
    monkeypatch.setattr("rag.insight_extraction.get_embeddings", lambda texts: [[0.3] * 4096])
    monkeypatch.setattr(
        "rag.insight_extraction.upsert_insight",
        lambda conn, content, emb: ("existing-uuid", False),
    )

    from rag.insight_extraction import extract_and_store_insights
    conn = MagicMock()
    driver = MagicMock()

    result = extract_and_store_insights(
        conn, driver, "source-1", [("chunk-1", "content 1")]
    )

    assert result["insights_reused"] == 1
    assert result["insights_extracted"] == 0
    related_call = next(payload for event, *payload in events if event == "related")
    new_insights_arg = related_call[1]
    assert new_insights_arg == []  # no new insights to link


def test_extract_and_store_insights_missing_api_key_skips_gracefully(monkeypatch):
    """KTD6b: an unset OPENCODE_API_KEY short-circuits before the extraction
    fan-out with a skipped marker; zero chunks are marked failed and the
    failure-rate gate is never invoked."""
    monkeypatch.setattr("rag.insight_extraction.settings.OPENCODE_API_KEY", "")
    parallel_mock = MagicMock()
    monkeypatch.setattr("rag.insight_extraction._extract_chunk_insights_parallel", parallel_mock)

    from rag.insight_extraction import extract_and_store_insights
    conn = MagicMock()
    driver = MagicMock()
    events = []

    result = extract_and_store_insights(
        conn, driver, "source-1", [("chunk-1", "content 1")],
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    assert result["skipped"] is True
    assert result["failed_chunks"] == []
    assert result["chunks_processed"] == 0
    parallel_mock.assert_not_called()
    assert ("store_start", {"total": 0}) in events
    assert ("store_done", {"total": 0}) in events


def test_extract_and_store_insights_reports_storage_progress(monkeypatch):
    events = _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        "rag.insight_extraction._extract_chunk_insights_parallel",
        lambda rows, progress_callback=None: (
            [
                ("chunk-1", "content 1", [{"insight": "insight A", "topics": []}]),
                ("chunk-2", "content 2", []),
            ],
            [],
        ),
    )
    monkeypatch.setattr("rag.insight_extraction.get_embeddings", lambda texts: [[0.1] * 4096])
    progress_events = []

    from rag.insight_extraction import extract_and_store_insights
    result = extract_and_store_insights(
        MagicMock(),
        MagicMock(),
        "source-1",
        [("chunk-1", "content 1"), ("chunk-2", "content 2")],
        progress_callback=lambda event, payload: progress_events.append((event, payload)),
    )

    assert result["chunks_processed"] == 2
    assert ("store_start", {"total": 2}) in progress_events
    assert ("store_done", {"total": 2}) in progress_events


def test_extract_and_store_insights_skips_blank_insight(monkeypatch):
    monkeypatch.setattr("rag.insight_extraction.settings.OPENCODE_API_KEY", "test-key")
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
    assert result["failed_chunks"] == []
    mock_embeddings.assert_not_called()
