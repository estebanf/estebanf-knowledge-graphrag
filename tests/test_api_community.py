from unittest.mock import patch

from fastapi.testclient import TestClient

from rag.api.main import create_app


def _client():
    return TestClient(create_app())


def _result():
    return {
        "metadata": {
            "scope_mode": "ids", "source_count": 2, "sources_excluded": [],
            "parameters": {
                "semantic_threshold": 0.85, "source_cooc_weight": 0.1,
                "cutoff": 0.5, "min_community_size": 3, "top_k_chunks": 5,
            },
        },
        "communities": [{
            "community_id": "0", "is_cross_source": True, "entity_count": 4,
            "entities": [{"entity_id": "e1", "canonical_name": "Alpha", "entity_type": "ORG"}],
            "contributing_sources": [{"source_id": "s1", "source_name": "Doc 1"}],
            "chunks": [{
                "chunk_id": "c1", "source_id": "s1", "source_name": "Doc 1",
                "entity_overlap_count": 3, "score": 2.25, "content": "content",
            }],
            "summary": "",
        }],
    }


@patch("rag.api.routes.community.detect_communities")
def test_community_endpoint_ids_mode(mock_detect):
    mock_detect.return_value = _result()
    response = _client().post("/api/community", json={"scope_mode": "ids", "source_ids": ["s1", "s2"]})
    assert response.status_code == 200
    assert response.json()["metadata"]["scope_mode"] == "ids"
    mock_detect.assert_called_once()


@patch("rag.api.routes.community.detect_communities")
def test_community_endpoint_rejects_invalid_scope_mode(mock_detect):
    response = _client().post("/api/community", json={"scope_mode": "invalid"})
    assert response.status_code == 422
    mock_detect.assert_not_called()


@patch("rag.api.routes.community.detect_communities")
def test_community_endpoint_search_mode_passes_options(mock_detect):
    mock_detect.return_value = _result()
    response = _client().post("/api/community", json={
        "scope_mode": "search",
        "criteria": ["AI trends"],
        "search_options": {"limit": 5, "min_score": 0.6},
        "community_options": {"semantic_threshold": 0.9},
    })
    assert response.status_code == 200
    kwargs = mock_detect.call_args[1]
    assert kwargs["scope_mode"] == "search"
    assert kwargs["search_options"] == {"limit": 5, "min_score": 0.6}
    assert kwargs["semantic_threshold"] == 0.9


@patch("rag.api.routes.community.detect_communities")
def test_community_endpoint_retrieve_mode_passes_options(mock_detect):
    mock_detect.return_value = _result()
    response = _client().post("/api/community", json={
        "scope_mode": "retrieve",
        "criteria": ["neural networks"],
        "retrieve_options": {"seed_count": 8, "result_count": 3},
    })
    assert response.status_code == 200
    kwargs = mock_detect.call_args[1]
    assert kwargs["retrieve_options"]["seed_count"] == 8
    assert kwargs["retrieve_options"]["result_count"] == 3


@patch("rag.api.routes.community.detect_communities")
def test_community_endpoint_passes_summarize_model(mock_detect):
    mock_detect.return_value = _result()
    response = _client().post("/api/community", json={
        "scope_mode": "ids", "source_ids": ["s1"],
        "summarize_model": "google/gemma-3-4b-it",
    })
    assert response.status_code == 200
    assert mock_detect.call_args[1]["summarize_model"] == "google/gemma-3-4b-it"
