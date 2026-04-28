import math
from unittest.mock import MagicMock, patch

import pytest


# --- _cosine_similarity ---

def test_cosine_similarity_identical_vectors():
    from rag.community import _cosine_similarity
    v = [1.0, 0.0, 1.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)

def test_cosine_similarity_orthogonal_vectors():
    from rag.community import _cosine_similarity
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

def test_cosine_similarity_zero_vector_returns_zero():
    from rag.community import _cosine_similarity
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# --- _resolve_scope ---

def test_resolve_scope_ids_returns_deduped():
    from rag.community import _resolve_scope
    result = _resolve_scope("ids", ["s1", "s2", "s1"], [], {}, {}, {})
    assert result == ["s1", "s2"]

@patch("rag.community.hybrid_search")
def test_resolve_scope_search_unions_source_ids(mock_search):
    from rag.community import _resolve_scope
    c1 = MagicMock(); c1.source_id = "src-A"
    c2 = MagicMock(); c2.source_id = "src-B"
    mock_search.side_effect = [[c1], [c2]]
    result = _resolve_scope("search", [], ["q1", "q2"], {}, {"limit": 5, "min_score": 0.0}, {})
    assert set(result) == {"src-A", "src-B"}


@patch("rag.community.resolve_retrieval_scope")
def test_resolve_scope_retrieve_uses_lightweight_retrieval_scope(mock_scope):
    from rag.community import _resolve_scope

    mock_scope.side_effect = [["src-A", "src-B"], ["src-B", "src-C"]]

    result = _resolve_scope(
        "retrieve",
        [],
        ["q1", "q2"],
        {"kind": "report"},
        {},
        {"seed_count": 3, "trace": False},
    )

    assert result == ["src-A", "src-B", "src-C"]
    assert mock_scope.call_count == 2

def test_resolve_scope_unknown_mode_raises():
    from rag.community import _resolve_scope
    with pytest.raises(ValueError, match="Unknown scope_mode"):
        _resolve_scope("unknown", [], [], {}, {}, {})


# --- _load_graph_data ---

@patch("rag.community.get_graph_driver")
@patch("rag.community.get_connection")
def test_load_graph_data_returns_entities_and_chunk_map(mock_conn, mock_graph):
    from rag.community import _load_graph_data
    conn = mock_conn.return_value.__enter__.return_value
    conn.execute.return_value.fetchall.return_value = [
        ("chunk-1", "source-1"), ("chunk-2", "source-1"),
    ]
    session = mock_graph.return_value.__enter__.return_value.session.return_value.__enter__.return_value
    session.run.return_value.data.return_value = [
        {"cid": "chunk-1", "entity_id": "e1", "canonical_name": "Alpha", "entity_type": "ORG"},
        {"cid": "chunk-2", "entity_id": "e2", "canonical_name": "Beta", "entity_type": "PERSON"},
    ]
    entities, chunk_to_source, excluded = _load_graph_data(["source-1"])
    assert "e1" in entities
    assert "e2" in entities
    assert chunk_to_source == {"chunk-1": "source-1", "chunk-2": "source-1"}
    assert excluded == []

@patch("rag.community.get_graph_driver")
@patch("rag.community.get_connection")
def test_load_graph_data_excludes_sources_with_no_entities(mock_conn, mock_graph):
    from rag.community import _load_graph_data
    conn = mock_conn.return_value.__enter__.return_value
    conn.execute.return_value.fetchall.return_value = [("chunk-1", "source-1")]
    session = mock_graph.return_value.__enter__.return_value.session.return_value.__enter__.return_value
    session.run.return_value.data.return_value = []
    entities, chunk_to_source, excluded = _load_graph_data(["source-1", "source-2"])
    assert "source-1" in excluded
    assert "source-2" in excluded


# --- _build_igraph ---

def test_build_igraph_creates_chunk_co_occurrence_edges():
    from rag.community import EntityNode, _build_igraph
    entities = {
        "e1": EntityNode("e1", "Alpha", "ORG", source_ids={"s1"}, chunk_ids={"c1"}),
        "e2": EntityNode("e2", "Beta", "PERSON", source_ids={"s1"}, chunk_ids={"c1"}),
    }
    with patch("rag.community._load_entity_embeddings", return_value={}), \
         patch("rag.community._load_cross_source_semantic_edges", return_value={}):
        g = _build_igraph(entities, {"c1": "s1"}, semantic_threshold=0.85, source_cooc_weight=0.1)
    assert g.vcount() == 2
    assert g.ecount() == 1
    assert g.es[0]["weight"] > 0

def test_build_igraph_adds_semantic_cross_source_edge():
    from rag.community import EntityNode, _build_igraph
    entities = {
        "e1": EntityNode("e1", "Alpha", "ORG", source_ids={"s1"}, chunk_ids={"c1"}),
        "e2": EntityNode("e2", "Beta", "PERSON", source_ids={"s2"}, chunk_ids={"c2"}),
    }
    cross_edge = {(0, 1): 0.5}
    with patch("rag.community._load_entity_embeddings", return_value={}), \
         patch("rag.community._load_cross_source_semantic_edges", return_value=cross_edge):
        g = _build_igraph(entities, {"c1": "s1", "c2": "s2"}, semantic_threshold=0.5, source_cooc_weight=0.1)
    assert g.ecount() == 1
    assert g.es[0]["weight"] == pytest.approx(0.5)


# --- _load_cross_source_semantic_edges ---

def test_cross_source_semantic_edges_creates_edge_above_threshold():
    from rag.community import EntityNode, _load_cross_source_semantic_edges
    entities = {
        "e1": EntityNode("e1", "Alpha", "ORG", source_ids={"s1"}, chunk_ids={"c1"}),
        "e2": EntityNode("e2", "Beta", "PERSON", source_ids={"s2"}, chunk_ids={"c2"}),
    }
    idx = {"e1": 0, "e2": 1}
    embeddings = {"e1": [1.0, 0.0], "e2": [1.0, 0.0]}

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)
    # ANN query for e1 returns e2 with sim=0.95
    mock_conn.execute.return_value.fetchall.return_value = [("e2", 0.95)]

    with patch("rag.community.get_connection", return_value=mock_conn):
        result = _load_cross_source_semantic_edges(
            entities, idx, embeddings,
            semantic_threshold=0.85, top_k=5, max_queries=100,
        )

    assert (0, 1) in result
    assert result[(0, 1)] == pytest.approx(0.95 * 0.5)


def test_cross_source_semantic_edges_threshold_gates_edge():
    from rag.community import EntityNode, _load_cross_source_semantic_edges
    entities = {
        "e1": EntityNode("e1", "Alpha", "ORG", source_ids={"s1"}, chunk_ids={"c1"}),
        "e2": EntityNode("e2", "Beta", "PERSON", source_ids={"s2"}, chunk_ids={"c2"}),
    }
    idx = {"e1": 0, "e2": 1}
    embeddings = {"e1": [1.0, 0.0], "e2": [0.5, 0.5]}

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)
    # ANN query returns e2 with sim below threshold
    mock_conn.execute.return_value.fetchall.return_value = [("e2", 0.70)]

    with patch("rag.community.get_connection", return_value=mock_conn):
        result = _load_cross_source_semantic_edges(
            entities, idx, embeddings,
            semantic_threshold=0.85, top_k=5, max_queries=100,
        )

    assert result == {}


def test_cross_source_semantic_edges_budget_cap_prioritizes_most_mentioned():
    from rag.community import EntityNode, _load_cross_source_semantic_edges

    # Build 10 entities; set chunk_ids so e0 has the most, e9 the fewest
    entities = {
        f"e{i}": EntityNode(f"e{i}", f"E{i}", "ORG", source_ids={f"s{i}"}, chunk_ids=set(f"c{j}" for j in range(10 - i)))
        for i in range(10)
    }
    idx = {eid: i for i, eid in enumerate(entities)}
    embeddings = {eid: [1.0, 0.0] for eid in entities}

    queried_ids: list[str] = []

    def fake_execute(sql, params):
        queried_ids.append(params[2])  # third param is the a_id being queried
        m = MagicMock()
        m.fetchall.return_value = []
        return m

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.side_effect = fake_execute

    with patch("rag.community.get_connection", return_value=mock_conn):
        _load_cross_source_semantic_edges(
            entities, idx, embeddings,
            semantic_threshold=0.85, top_k=5, max_queries=3,
        )

    assert len(queried_ids) == 3
    # The 3 queried entities must be the ones with the most chunk_ids (e0, e1, e2)
    assert set(queried_ids) == {"e0", "e1", "e2"}


def test_build_igraph_cross_source_does_not_overwrite_chunk_cooc_edge():
    from rag.community import EntityNode, _build_igraph
    # e1 and e2 share a chunk AND a cross-source semantic edge should be ignored
    entities = {
        "e1": EntityNode("e1", "Alpha", "ORG", source_ids={"s1"}, chunk_ids={"c1"}),
        "e2": EntityNode("e2", "Beta", "PERSON", source_ids={"s2"}, chunk_ids={"c1"}),
    }
    # The chunk-cooc edge produces weight > 0; semantic would produce 0.45
    semantic_edges = {(0, 1): 0.45}
    with patch("rag.community._load_entity_embeddings", return_value={}), \
         patch("rag.community._load_cross_source_semantic_edges", return_value=semantic_edges):
        g = _build_igraph(entities, {"c1": "s1"}, semantic_threshold=0.85, source_cooc_weight=0.1)

    assert g.ecount() == 1
    # The weight must be the chunk-cooc weight (base from 1/sqrt(1*1)=1.0 * 1.5 cross factor),
    # NOT the semantic weight of 0.45
    assert g.es[0]["weight"] > 0.45


# --- _run_leiden ---

def test_run_leiden_returns_communities_meeting_min_size():
    import igraph
    from rag.community import _run_leiden
    g = igraph.Graph(n=6, directed=False)
    g.vs["entity_id"] = ["e1", "e2", "e3", "e4", "e5", "e6"]
    g.add_edges([(0,1),(1,2),(0,2),(3,4),(4,5),(3,5)])
    g.es["weight"] = [1.0] * 6
    communities = _run_leiden(g, min_community_size=3)
    assert len(communities) >= 1
    assert all(len(c) >= 3 for c in communities)

def test_run_leiden_empty_graph_returns_empty():
    import igraph
    from rag.community import _run_leiden
    g = igraph.Graph(n=0, directed=False)
    g.vs["entity_id"] = []
    assert _run_leiden(g, min_community_size=3) == []


# --- _score_and_select_chunks ---

@patch("rag.community._expand_chunk_texts", return_value={})
@patch("rag.community.get_connection")
def test_score_chunks_drops_below_cutoff(mock_conn, mock_expand):
    from rag.community import EntityNode, _score_and_select_chunks
    conn = mock_conn.return_value.__enter__.return_value
    conn.execute.return_value.fetchall.return_value = [
        ("chunk-1", "text A"), ("chunk-2", "text B"),
    ]
    entities = {
        "e1": EntityNode("e1", "A", "ORG", source_ids={"s1"}, chunk_ids={"chunk-1"}),
        "e2": EntityNode("e2", "B", "ORG", source_ids={"s1"}, chunk_ids={"chunk-2"}),
        "e3": EntityNode("e3", "C", "ORG", source_ids={"s1"}, chunk_ids={"chunk-2"}),
        "e4": EntityNode("e4", "D", "ORG", source_ids={"s1"}, chunk_ids={"chunk-2"}),
    }
    # chunk-1: overlap=1, score=1/4=0.25 < 0.5 → excluded
    # chunk-2: overlap=3, score=9/4=2.25 → included
    results = _score_and_select_chunks(
        ["e1","e2","e3","e4"], entities, {"chunk-1":"s1","chunk-2":"s1"},
        {"s1": "S1"}, cutoff=0.5, top_k=5
    )
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-2"

@patch("rag.community._expand_chunk_texts", return_value={})
@patch("rag.community.get_connection")
def test_score_chunks_round_robin_for_cross_source(mock_conn, mock_expand):
    from rag.community import EntityNode, _score_and_select_chunks
    conn = mock_conn.return_value.__enter__.return_value
    conn.execute.return_value.fetchall.return_value = [
        ("c1", "s1 text"), ("c2", "s2 text"), ("c3", "s1 text2"),
    ]
    entities = {
        "e1": EntityNode("e1", "A", "ORG", source_ids={"s1"}, chunk_ids={"c1","c3"}),
        "e2": EntityNode("e2", "B", "ORG", source_ids={"s2"}, chunk_ids={"c2"}),
    }
    results = _score_and_select_chunks(
        ["e1","e2"], entities, {"c1":"s1","c2":"s2","c3":"s1"},
        {"s1":"S1","s2":"S2"}, cutoff=0.0, top_k=3
    )
    source_ids = [r.source_id for r in results]
    assert "s2" in source_ids  # round-robin ensures both sources represented


@patch("rag.community._expand_chunk_texts")
@patch("rag.community.get_connection")
def test_score_chunks_expands_content_with_shared_retrieval_logic(mock_conn, mock_expand):
    from rag.community import EntityNode, _score_and_select_chunks
    conn = mock_conn.return_value.__enter__.return_value
    conn.execute.return_value.fetchall.return_value = [("c1", "short content")]
    mock_expand.return_value = {"c1": "expanded content"}
    entities = {
        "e1": EntityNode("e1", "A", "ORG", source_ids={"s1"}, chunk_ids={"c1"}),
    }

    results = _score_and_select_chunks(
        ["e1"], entities, {"c1": "s1"}, {"s1": "S1"}, cutoff=0.0, top_k=5
    )

    assert results[0].content == "expanded content"


# --- _summarize_community ---

@patch("rag.community.requests.post")
def test_summarize_community_calls_openrouter(mock_post, monkeypatch):
    from rag.community import Community, ChunkResult, ContributingSource, _summarize_community
    monkeypatch.setattr("rag.community.settings.OPENROUTER_API_KEY", "test-key")
    mock_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "Summary text"}}]
    }
    community = Community(
        community_id="0", is_cross_source=False, entity_count=2,
        entities=[{"entity_id": "e1", "canonical_name": "Alpha", "entity_type": "ORG"}],
        contributing_sources=[ContributingSource("s1", "Doc 1")],
        chunks=[ChunkResult("c1", "s1", "Doc 1", 1, 1.0, "chunk content")],
    )
    result = _summarize_community(community, "google/gemma-3-4b-it")
    assert result == "Summary text"

def test_summarize_community_raises_without_api_key(monkeypatch):
    from rag.community import Community, _summarize_community
    monkeypatch.setattr("rag.community.settings.OPENROUTER_API_KEY", "")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        _summarize_community(Community("0", False, 0, [], [], []), "model")


# --- detect_communities integration ---

@patch("rag.community._summarize_community")
@patch("rag.community._score_and_select_chunks")
@patch("rag.community._run_leiden")
@patch("rag.community._build_igraph")
@patch("rag.community._load_source_names")
@patch("rag.community._load_graph_data")
@patch("rag.community._resolve_scope")
def test_detect_communities_returns_correct_shape(
    mock_scope, mock_graph_data, mock_names, mock_build,
    mock_leiden, mock_chunks, mock_summarize
):
    from rag.community import EntityNode, ChunkResult, detect_communities
    mock_scope.return_value = ["s1"]
    entities = {
        "e1": EntityNode("e1", "Alpha", "ORG", source_ids={"s1"}, chunk_ids={"c1"}),
        "e2": EntityNode("e2", "Beta", "PERSON", source_ids={"s1"}, chunk_ids={"c1"}),
        "e3": EntityNode("e3", "Gamma", "ORG", source_ids={"s1"}, chunk_ids={"c1"}),
    }
    mock_graph_data.return_value = (entities, {"c1": "s1"}, [])
    mock_names.return_value = {"s1": "Source One"}
    mock_build.return_value = MagicMock()
    mock_leiden.return_value = [["e1", "e2", "e3"]]
    mock_chunks.return_value = [ChunkResult("c1", "s1", "Source One", 3, 1.0, "content")]

    result = detect_communities(
        scope_mode="ids", source_ids=["s1"], criteria=[], filters={},
        search_options={}, retrieve_options={}
    )
    assert result["metadata"]["scope_mode"] == "ids"
    assert result["metadata"]["source_count"] == 1
    assert len(result["communities"]) == 1
    assert result["communities"][0]["entity_count"] == 3
    params = result["metadata"]["parameters"]
    assert "cross_source_top_k" in params
    assert "max_cross_source_queries" in params


def test_detect_communities_new_params_override_defaults():
    from rag.community import detect_communities
    with patch("rag.community._resolve_scope", return_value=[]), \
         patch("rag.community._load_graph_data", return_value=({}, {}, [])), \
         patch("rag.community._load_source_names", return_value={}), \
         patch("rag.community._build_igraph") as mock_build, \
         patch("rag.community._run_leiden", return_value=[]):
        mock_build.return_value = MagicMock()
        result = detect_communities(
            scope_mode="ids", source_ids=[], criteria=[], filters={},
            search_options={}, retrieve_options={},
            cross_source_top_k=3, max_cross_source_queries=99,
        )
        params = result["metadata"]["parameters"]
        assert params["cross_source_top_k"] == 3
        assert params["max_cross_source_queries"] == 99
