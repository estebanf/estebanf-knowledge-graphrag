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


def test_upsert_insight_creates_new_when_below_threshold():
    from rag.insight_extraction import upsert_insight
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.side_effect = [("existing-uuid", 0.80), ("new-uuid",)]
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    insight_id, is_new = upsert_insight(conn, "different insight", [0.9] * 4096)
    assert is_new is True


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


def test_link_related_insights_creates_mutual_edges():
    from rag.insight_extraction import link_related_insights
    conn = MagicMock()
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = lambda s: session
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    cursor = MagicMock()
    b_emb = [0.2] * 4096
    cursor.fetchall.side_effect = [
        [("b-id", 0.88, b_emb)],            # A's neighbors
        [("a-id", 0.88, [0.1]*4096)],       # B's neighbors — includes A, so mutual
    ]
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    link_related_insights(conn, driver, "source-a", "a-id", [0.1] * 4096)
    assert session.run.call_count == 1
    cypher = session.run.call_args[0][0]
    assert "RELATED_TO" in cypher


def test_link_related_insights_excludes_current_source_candidates():
    from rag.insight_extraction import link_related_insights
    conn = MagicMock()
    driver = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    link_related_insights(conn, driver, "source-a", "a-id", [0.1] * 4096)

    sql = cursor.execute.call_args.args[0]
    params = cursor.execute.call_args.args[1]
    assert "NOT EXISTS" in sql
    assert "c.source_id = %s" in sql
    assert params[1] == "a-id"
    assert params[2] == "source-a"


def test_link_related_insights_skips_non_mutual():
    from rag.insight_extraction import link_related_insights
    conn = MagicMock()
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = lambda s: session
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    cursor = MagicMock()
    cursor.fetchall.side_effect = [
        [("b-id", 0.88, [0.2]*4096)],   # A's neighbors: B
        [("c-id", 0.90, [0.3]*4096)],   # B's neighbors: C (not A) — not mutual
    ]
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    link_related_insights(conn, driver, "source-a", "a-id", [0.1] * 4096)
    session.run.assert_not_called()


def test_link_related_insights_accepts_database_vector_strings():
    from rag.insight_extraction import link_related_insights
    conn = MagicMock()
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = lambda s: session
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    cursor = MagicMock()
    cursor.fetchall.side_effect = [
        [("b-id", 0.88, "[0.2,0.2]")],
        [("a-id",)],
    ]
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    link_related_insights(conn, driver, "source-a", "a-id", [0.1] * 2)

    assert cursor.execute.call_args_list[1].args[1][2] == "[0.2,0.2]"
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
