import json
from unittest.mock import patch

from typer.testing import CliRunner

from rag.cli import app

runner = CliRunner()


def _result() -> dict:
    return {
        "metadata": {
            "scope_mode": "ids", "source_count": 1, "sources_excluded": [],
            "parameters": {
                "semantic_threshold": 0.85, "source_cooc_weight": 0.1,
                "cutoff": 0.5, "min_community_size": 3, "top_k_chunks": 5,
            },
        },
        "communities": [{
            "community_id": "0", "is_cross_source": False, "entity_count": 3,
            "entities": [], "contributing_sources": [], "chunks": [], "summary": "",
        }],
    }


@patch("rag.cli.detect_communities")
def test_community_ids_basic(mock_detect):
    mock_detect.return_value = _result()
    result = runner.invoke(app, ["community", "ids", "source-1", "source-2"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["metadata"]["scope_mode"] == "ids"
    mock_detect.assert_called_once_with(
        scope_mode="ids", source_ids=["source-1", "source-2"], criteria=[], filters={},
        search_options={}, retrieve_options={},
        semantic_threshold=None, cutoff=None, min_community_size=None,
        top_k_chunks=None, summarize_model=None,
        cross_source_top_k=None, max_cross_source_queries=None,
    )


@patch("rag.cli.detect_communities")
def test_community_ids_with_overrides(mock_detect):
    mock_detect.return_value = _result()
    result = runner.invoke(app, [
        "community", "ids", "source-1",
        "--semantic-threshold", "0.9", "--cutoff", "0.3",
        "--min-community-size", "5", "--top-k", "3",
        "--summarize", "google/gemma-3-4b-it",
    ])
    assert result.exit_code == 0
    mock_detect.assert_called_once_with(
        scope_mode="ids", source_ids=["source-1"], criteria=[], filters={},
        search_options={}, retrieve_options={},
        semantic_threshold=0.9, cutoff=0.3, min_community_size=5,
        top_k_chunks=3, summarize_model="google/gemma-3-4b-it",
        cross_source_top_k=None, max_cross_source_queries=None,
    )


@patch("rag.cli.detect_communities")
def test_community_search_passes_criteria(mock_detect):
    mock_detect.return_value = _result()
    result = runner.invoke(app, [
        "community", "search", "machine learning", "neural networks",
        "--limit", "5", "--min-score", "0.6",
    ])
    assert result.exit_code == 0
    kwargs = mock_detect.call_args[1]
    assert kwargs["scope_mode"] == "search"
    assert kwargs["criteria"] == ["machine learning", "neural networks"]
    assert kwargs["search_options"] == {"limit": 5, "min_score": 0.6}


@patch("rag.cli.detect_communities")
def test_community_search_rejects_invalid_filter(mock_detect):
    result = runner.invoke(app, ["community", "search", "query", "--filter", "badformat"])
    assert result.exit_code == 1
    mock_detect.assert_not_called()


@patch("rag.cli.detect_communities")
def test_community_retrieve_passes_options(mock_detect):
    mock_detect.return_value = _result()
    result = runner.invoke(app, [
        "community", "retrieve", "AI trends",
        "--seed-count", "8", "--result-count", "4",
    ])
    assert result.exit_code == 0
    kwargs = mock_detect.call_args[1]
    assert kwargs["scope_mode"] == "retrieve"
    assert kwargs["criteria"] == ["AI trends"]
    assert kwargs["retrieve_options"]["seed_count"] == 8
    assert kwargs["retrieve_options"]["result_count"] == 4
