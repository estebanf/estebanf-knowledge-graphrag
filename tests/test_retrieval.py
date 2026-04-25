from unittest.mock import MagicMock

import pytest

from rag.retrieval import (
    EntityCandidate,
    RetrievalCandidate,
    TraceLogger,
    _apply_expanded_chunk_text,
    _expand_chunk_texts,
    _expand_neighbor_contexts,
    _load_chunk_ids_for_entity,
    _load_seed_entities,
    aggregate_root_score,
    expand_seed_candidate,
    normalize_query_variants,
    retrieve,
    weighted_reciprocal_rank_fusion,
)


def test_normalize_query_variants_drops_empty_and_duplicate_values():
    variants = normalize_query_variants(
        {
            "original": "What changed in the policy?",
            "hyde": "Hypothetical answer",
            "expanded": "What changed in the policy?",
            "step_back": "  ",
            "decomposed": ["What changed in the policy?", "Who changed it?", ""],
        }
    )

    assert variants["original"] == "What changed in the policy?"
    assert variants["hyde"] == "Hypothetical answer"
    assert "expanded" not in variants
    assert "step_back" not in variants
    assert variants["decomposed"] == ["Who changed it?"]


def test_weighted_reciprocal_rank_fusion_merges_by_chunk_id():
    dense = [
        RetrievalCandidate(chunk_id="chunk-1", chunk="alpha", source_id="s1", source_path="/a", source_metadata={}, score=0.9),
        RetrievalCandidate(chunk_id="chunk-2", chunk="beta", source_id="s1", source_path="/a", source_metadata={}, score=0.8),
    ]
    sparse = [
        RetrievalCandidate(chunk_id="chunk-2", chunk="beta", source_id="s1", source_path="/a", source_metadata={}, score=12.0),
        RetrievalCandidate(chunk_id="chunk-3", chunk="gamma", source_id="s2", source_path="/b", source_metadata={}, score=10.0),
    ]

    fused = weighted_reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse},
        rrf_k=60,
        weights={"dense": 1.0, "sparse": 0.5},
        score_floor=0.0,
    )

    assert [item.chunk_id for item in fused] == ["chunk-2", "chunk-1", "chunk-3"]
    assert fused[0].score > fused[1].score > fused[2].score


def test_aggregate_root_score_uses_component_weights_and_bonus():
    score = aggregate_root_score(
        root_score=0.8,
        first_hop_scores=[0.4, 0.6],
        second_hop_scores=[0.1, 0.2],
        root_weight=0.6,
        first_hop_weight=0.25,
        second_hop_weight=0.15,
        multi_path_bonus=0.05,
    )

    assert score == 0.6 * 0.8 + 0.25 * 0.6 + 0.15 * 0.2 + 0.05


def test_retrieve_emits_trace_and_returns_final_results(monkeypatch):
    traces: list[str] = []

    monkeypatch.setattr(
        "rag.retrieval.generate_query_variants",
        lambda query, trace_logger=None: (
            trace_logger.emit("generated query variants")
            or {
                "original": query,
                "hyde": "Synthetic answer",
                "decomposed": ["part one"],
            }
        ),
    )
    monkeypatch.setattr(
        "rag.retrieval.run_first_stage_retrieval",
        lambda **kwargs: [
            RetrievalCandidate(
                chunk_id="chunk-1",
                chunk="Root chunk",
                source_id="source-1",
                source_path="/tmp/source-1.md",
                source_metadata={"kind": "report"},
                score=0.5,
            )
        ],
    )
    monkeypatch.setattr(
        "rag.retrieval.rerank_candidates",
        lambda query, candidates, top_n, trace_logger=None: [
            RetrievalCandidate(
                chunk_id="chunk-1",
                chunk="Root chunk",
                source_id="source-1",
                source_path="/tmp/source-1.md",
                source_metadata={"kind": "report"},
                score=0.95,
            )
        ],
    )
    monkeypatch.setattr(
        "rag.retrieval.expand_seed_candidate",
        lambda seed, query, source_ids, filters, entity_confidence_threshold,
               first_hop_similarity_threshold, second_hop_similarity_threshold,
               trace_logger=None, **kwargs: {
            "score": seed.score,
            "chunk": seed.chunk,
            "chunk_id": seed.chunk_id,
            "source_id": seed.source_id,
            "source_path": seed.source_path,
            "source_metadata": seed.source_metadata,
            "related": [],
            "_root_score": seed.score,
            "_first_hop_scores": [],
            "_second_hop_scores": [],
            "_multi_path_bonus": 0.0,
        },
    )
    monkeypatch.setattr(
        "rag.retrieval.finalize_root_results",
        lambda query, root_results, result_count, trace_logger=None: root_results,
    )
    monkeypatch.setattr("rag.retrieval._expand_neighbor_contexts", lambda conn, results: None)

    response = retrieve(
        query="what happened",
        source_ids=["source-1"],
        filters={"kind": "report"},
        seed_count=3,
        result_count=2,
        rrf_k=40,
        entity_confidence_threshold=0.8,
        first_hop_similarity_threshold=0.6,
        second_hop_similarity_threshold=0.7,
        trace=True,
        trace_printer=traces.append,
    )

    assert response["retrieval_results"][0]["chunk_id"] == "chunk-1"
    assert any("starting retrieval" in entry for entry in traces)
    assert any("generated query variants" in entry for entry in traces)
    assert any("selected 1 seed chunks" in entry for entry in traces)


def test_trace_logger_emits_only_when_enabled():
    entries: list[str] = []
    logger = TraceLogger(enabled=True, printer=entries.append)
    logger.emit("hello")

    disabled_entries: list[str] = []
    disabled_logger = TraceLogger(enabled=False, printer=disabled_entries.append)
    disabled_logger.emit("hidden")

    assert entries == ["hello"]
    assert disabled_entries == []


def test_expand_seed_candidate_uses_per_seed_wall_clock_budget(monkeypatch):
    seed = RetrievalCandidate(
        chunk_id="seed-1",
        chunk="seed chunk",
        source_id="source-1",
        source_path="/tmp/source.md",
        source_metadata={},
        score=0.9,
    )

    class _Time:
        def __init__(self):
            self.values = iter([10.0, 10.0, 15.1])

        def monotonic(self):
            return next(self.values)

    monkeypatch.setattr("rag.retrieval.time", _Time())
    monkeypatch.setattr("rag.retrieval.settings.RETRIEVAL_MAX_GRAPH_EXPANSION_MS_PER_SEED", 4000)
    monkeypatch.setattr("rag.retrieval._load_seed_entities", lambda driver, chunk_id: [])

    traces: list[str] = []
    result = expand_seed_candidate(
        seed,
        query="what happened",
        source_ids=[],
        filters={},
        entity_confidence_threshold=0.8,
        first_hop_similarity_threshold=0.5,
        second_hop_similarity_threshold=0.5,
        conn=object(),
        driver=object(),
        trace_logger=TraceLogger(enabled=True, printer=traces.append),
        budget={"llm_calls": 0},
    )

    assert result["chunk_id"] == "seed-1"
    assert not any("graph time budget exhausted before expanding seed" in entry for entry in traces)
    assert any("loaded 0 entities for seed seed-1" in entry for entry in traces)


def test_expand_seed_candidate_falls_back_to_same_source_neighbors(monkeypatch):
    seed = RetrievalCandidate(
        chunk_id="seed-1",
        chunk="seed chunk",
        source_id="source-1",
        source_path="/tmp/source.md",
        source_metadata={},
        score=0.9,
    )
    entity = EntityCandidate(entity_id="entity-1", name="agentic identity", entity_type="CONCEPT")
    fallback_chunk = RetrievalCandidate(
        chunk_id="neighbor-1",
        chunk="neighbor chunk",
        source_id="source-1",
        source_path="/tmp/source.md",
        source_metadata={},
        score=0.55,
    )

    monkeypatch.setattr("rag.retrieval.time.monotonic", lambda: 10.0)
    monkeypatch.setattr("rag.retrieval._load_seed_entities", lambda driver, chunk_id: [entity])
    monkeypatch.setattr("rag.retrieval._select_entity_names", lambda *args, **kwargs: ["agentic identity"])
    monkeypatch.setattr("rag.retrieval._generate_entity_query", lambda *args, **kwargs: "agentic identity security")
    monkeypatch.setattr(
        "rag.retrieval._load_chunk_ids_for_entity",
        lambda driver, entity_name, entity_type: ["seed-1"],
    )
    monkeypatch.setattr("rag.retrieval._fetch_chunk_candidates_by_ids", lambda *args, **kwargs: [seed])
    monkeypatch.setattr("rag.retrieval._load_entities_for_chunks", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "rag.retrieval._fetch_same_source_neighbor_candidates",
        lambda *args, **kwargs: [fallback_chunk],
    )

    traces: list[str] = []
    result = expand_seed_candidate(
        seed,
        query="what happened",
        source_ids=[],
        filters={},
        entity_confidence_threshold=0.8,
        first_hop_similarity_threshold=0.5,
        second_hop_similarity_threshold=0.5,
        conn=object(),
        driver=object(),
        trace_logger=TraceLogger(enabled=True, printer=traces.append),
        budget={"llm_calls": 0, "query_started_at": 10.0},
    )

    assert result["related"][0]["chunks"][0]["chunk_id"] == "neighbor-1"
    assert any("falling back to same-source neighbors for entity agentic identity" in entry for entry in traces)


def test_load_seed_entities_uses_mentions_only():
    class FakeSession:
        def __init__(self):
            self.query = None

        def run(self, query, **kwargs):
            self.query = query
            return []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeDriver:
        def __init__(self):
            self.session_obj = FakeSession()

        def session(self):
            return self.session_obj

    driver = FakeDriver()
    _load_seed_entities(driver, "chunk-1")

    assert "MENTIONS" in driver.session_obj.query
    assert "MENTIONED_IN" not in driver.session_obj.query


def test_load_chunk_ids_for_entity_matches_name_and_type_via_mentions():
    class FakeSession:
        def __init__(self):
            self.query = None

        def run(self, query, **kwargs):
            self.query = query
            self.kwargs = kwargs
            return []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeDriver:
        def __init__(self):
            self.session_obj = FakeSession()

        def session(self):
            return self.session_obj

    driver = FakeDriver()
    _load_chunk_ids_for_entity(driver, "Acme", "ORGANIZATION")

    assert "MENTIONS" in driver.session_obj.query
    assert "MENTIONED_IN" not in driver.session_obj.query
    assert "canonical_name" in driver.session_obj.query
    assert "entity_type" in driver.session_obj.query
    assert driver.session_obj.kwargs["entity_name"] == "Acme"
    assert driver.session_obj.kwargs["entity_type"] == "ORGANIZATION"


def test_fetch_chunk_candidates_by_ids_uses_provided_vector_without_reembedding(monkeypatch):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    called = False

    def _unexpected(_texts):
        nonlocal called
        called = True
        raise AssertionError("get_embeddings should not be called when vector is provided")

    monkeypatch.setattr("rag.retrieval.get_embeddings", _unexpected)

    from rag.retrieval import _fetch_chunk_candidates_by_ids

    _fetch_chunk_candidates_by_ids(
        conn,
        ["00000000-0000-0000-0000-000000000001"],
        "policy query",
        source_ids=[],
        filters={},
        limit=5,
        vector=[0.1, 0.2, 0.3],
    )

    assert called is False


def test_fetch_same_source_neighbor_candidates_uses_provided_vector_without_reembedding(monkeypatch):
    conn = MagicMock()
    conn.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=("source-1", 5))),
        MagicMock(fetchall=MagicMock(return_value=[])),
    ]
    called = False

    def _unexpected(_texts):
        nonlocal called
        called = True
        raise AssertionError("get_embeddings should not be called when vector is provided")

    monkeypatch.setattr("rag.retrieval.get_embeddings", _unexpected)

    from rag.retrieval import _fetch_same_source_neighbor_candidates

    seed = RetrievalCandidate(
        chunk_id="00000000-0000-0000-0000-000000000010",
        chunk="seed",
        source_id="source-1",
        source_path="/tmp/source.md",
        source_metadata={},
        score=0.9,
    )

    _fetch_same_source_neighbor_candidates(
        conn,
        seed,
        "policy query",
        source_ids=[],
        filters={},
        limit=5,
        vector=[0.1, 0.2, 0.3],
    )

    assert called is False


def test_expand_seed_candidate_uses_chunk_mediated_second_hop(monkeypatch):
    seed = RetrievalCandidate(
        chunk_id="seed-1",
        chunk="seed chunk",
        source_id="source-1",
        source_path="/tmp/source.md",
        source_metadata={},
        score=0.9,
    )
    entity1 = EntityCandidate(entity_id="entity-1", name="identity", entity_type="CONCEPT")
    entity2 = EntityCandidate(entity_id="entity-2", name="policy", entity_type="POLICY")
    first_hop_chunk = RetrievalCandidate(
        chunk_id="chunk-2",
        chunk="identity policy chunk",
        source_id="source-2",
        source_path="/tmp/source-2.md",
        source_metadata={},
        score=0.81,
    )
    second_hop_chunk = RetrievalCandidate(
        chunk_id="chunk-3",
        chunk="policy implementation chunk",
        source_id="source-3",
        source_path="/tmp/source-3.md",
        source_metadata={},
        score=0.77,
    )

    second_hop_queries = []

    monkeypatch.setattr("rag.retrieval.time.monotonic", lambda: 10.0)
    monkeypatch.setattr("rag.retrieval._load_seed_entities", lambda driver, chunk_id: [entity1])
    monkeypatch.setattr("rag.retrieval._select_entity_names", lambda *args, **kwargs: ["identity"])
    monkeypatch.setattr(
        "rag.retrieval._generate_entity_query",
        lambda query, seed, entity_name, **kwargs: (
            second_hop_queries.append((entity_name, kwargs)) or f"query for {entity_name}"
        ),
    )
    monkeypatch.setattr(
        "rag.retrieval._load_chunk_ids_for_entity",
        lambda driver, entity_name, entity_type: ["chunk-2"] if entity_name == "identity" else ["chunk-3"],
    )
    monkeypatch.setattr(
        "rag.retrieval._fetch_chunk_candidates_by_ids",
        lambda conn, chunk_ids, query_text, **kwargs: (
            [first_hop_chunk] if chunk_ids == ["chunk-2"] else [second_hop_chunk]
        ),
    )
    monkeypatch.setattr(
        "rag.retrieval._load_entities_for_chunks",
        lambda driver, chunk_ids: {"chunk-2": [entity1, entity2]},
    )
    monkeypatch.setattr(
        "rag.retrieval._select_second_hop_entities_from_chunks",
        lambda query, seed, entity_name, chunk_entity_map, trace_logger=None: [(first_hop_chunk, entity2)],
    )

    result = expand_seed_candidate(
        seed,
        query="what happened",
        source_ids=[],
        filters={},
        entity_confidence_threshold=0.8,
        first_hop_similarity_threshold=0.5,
        second_hop_similarity_threshold=0.5,
        conn=object(),
        driver=object(),
        budget={"llm_calls": 0, "query_started_at": 10.0},
    )

    assert result["related"][0]["second_level_related"][0]["entity"] == "policy"
    assert result["related"][0]["second_level_related"][0]["chunks"][0]["chunk_id"] == "chunk-3"
    assert second_hop_queries[1][0] == "policy"
    assert second_hop_queries[1][1]["entity_context"]["entity1"] == "identity"
    assert second_hop_queries[1][1]["entity_context"]["entity2"] == "policy"


def test_expand_neighbor_contexts_expands_roots_and_related_chunks(monkeypatch):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("root", "root-prev", "source-1", 0, "root previous"),
        ("root", "root", "source-1", 1, "root current"),
        ("root", "root-next", "source-1", 2, "root next"),
        ("related", "related", "source-1", 3, "related current"),
        ("related", "related-next", "source-1", 4, "related next"),
        ("second", "second-prev", "source-2", 8, "second previous"),
        ("second", "second", "source-2", 9, "second current"),
    ]
    counts = {
        "root current": 50,
        "root current\n\nroot next": 120,
        "root previous\n\nroot current\n\nroot next": 220,
        "related current": 50,
        "related current\n\nrelated next": 210,
        "second current": 50,
        "second previous\n\nsecond current": 210,
    }
    monkeypatch.setattr("rag.retrieval._token_count", lambda text: counts[text])
    monkeypatch.setattr("rag.retrieval.settings.RETRIEVAL_EXPANSION_MIN_TOKENS", 200)
    results = [
        {
            "chunk_id": "root",
            "chunk": "root current",
            "score": 0.9,
            "related": [
                {
                    "entity": "entity-1",
                    "chunks": [
                        {"chunk_id": "related", "chunk": "related current", "score": 0.6},
                    ],
                    "second_level_related": [
                        {
                            "entity": "entity-2",
                            "relationship": {"label": "X", "metadata": {}},
                            "chunks": [
                                {"chunk_id": "second", "chunk": "second current", "score": 0.5},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    _expand_neighbor_contexts(conn, results)

    assert results[0]["chunk"] == "root previous\n\nroot current\n\nroot next"
    assert results[0]["related"][0]["chunks"][0]["chunk"] == "related current\n\nrelated next"
    assert results[0]["related"][0]["second_level_related"][0]["chunks"][0]["chunk"] == (
        "second previous\n\nsecond current"
    )


def test_expand_neighbor_contexts_dedupes_chunk_lookups():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("root", "root", "source-1", 1, "root current"),
    ]
    results = [
        {
            "chunk_id": "root",
            "chunk": "root current",
            "score": 0.9,
            "related": [
                {
                    "entity": "entity-1",
                    "chunks": [
                        {"chunk_id": "root", "chunk": "root current", "score": 0.6},
                    ],
                    "second_level_related": [],
                }
            ],
        }
    ]

    _expand_neighbor_contexts(conn, results)

    conn.execute.assert_called_once()


def test_expand_chunk_texts_prefers_forward_then_backfills_until_min_tokens(monkeypatch):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("root", "root", "source-1", 5, "root"),
        ("root", "forward-1", "source-1", 6, "forward one"),
        ("root", "forward-2", "source-1", 7, "forward two"),
        ("root", "back-1", "source-1", 4, "back one"),
    ]
    counts = {
        "root": 50,
        "root\n\nforward one": 120,
        "root\n\nforward one\n\nforward two": 170,
        "back one\n\nroot\n\nforward one\n\nforward two": 230,
    }
    monkeypatch.setattr("rag.retrieval._token_count", lambda text: counts[text])
    monkeypatch.setattr("rag.retrieval.settings.RETRIEVAL_EXPANSION_MIN_TOKENS", 200)
    monkeypatch.setattr("rag.retrieval.settings.RETRIEVAL_EXPANSION_MAX_TOKENS", 600)

    expanded = _expand_chunk_texts(conn, ["root"])

    assert expanded["root"] == "back one\n\nroot\n\nforward one\n\nforward two"


def test_expand_chunk_texts_stops_when_max_tokens_would_be_exceeded(monkeypatch):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("root", "root", "source-1", 5, "root"),
        ("root", "forward-1", "source-1", 6, "forward one"),
        ("root", "forward-2", "source-1", 7, "forward two"),
        ("root", "back-1", "source-1", 4, "back one"),
    ]
    counts = {
        "root": 50,
        "root\n\nforward one": 120,
        "root\n\nforward one\n\nforward two": 170,
        "back one\n\nroot\n\nforward one\n\nforward two": 230,
    }
    monkeypatch.setattr("rag.retrieval._token_count", lambda text: counts[text])
    monkeypatch.setattr("rag.retrieval.settings.RETRIEVAL_EXPANSION_MIN_TOKENS", 200)
    monkeypatch.setattr("rag.retrieval.settings.RETRIEVAL_EXPANSION_MAX_TOKENS", 180)

    expanded = _expand_chunk_texts(conn, ["root"])

    assert expanded["root"] == "root\n\nforward one\n\nforward two"


def test_apply_expanded_chunk_text_updates_nested_results():
    expanded = {
        "root": "expanded root",
        "related": "expanded related",
        "second": "expanded second",
    }
    results = [
        {
            "chunk_id": "root",
            "chunk": "root current",
            "score": 0.9,
            "related": [
                {
                    "entity": "entity-1",
                    "chunks": [
                        {"chunk_id": "related", "chunk": "related current", "score": 0.6},
                    ],
                    "second_level_related": [
                        {
                            "entity": "entity-2",
                            "relationship": {"label": "X", "metadata": {}},
                            "chunks": [
                                {"chunk_id": "second", "chunk": "second current", "score": 0.5},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    _apply_expanded_chunk_text(results, expanded)

    assert results[0]["chunk"] == "expanded root"
    assert results[0]["related"][0]["chunks"][0]["chunk"] == "expanded related"
    assert results[0]["related"][0]["second_level_related"][0]["chunks"][0]["chunk"] == "expanded second"
