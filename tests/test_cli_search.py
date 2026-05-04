import json
from unittest.mock import patch

from typer.testing import CliRunner

from rag.cli import app
from rag.retrieval import HybridSearchResults, RetrievalCandidate

runner = CliRunner()


def _search_results() -> HybridSearchResults:
    return HybridSearchResults(
        chunks=[
            RetrievalCandidate(
                chunk_id="chunk-1",
                chunk="some chunk text",
                source_id="source-1",
                source_path="/tmp/doc.md",
                source_metadata={"kind": "report"},
                score=0.85,
            )
        ],
        insights=[],
    )


@patch("rag.cli.hybrid_search")
def test_search_command_prints_json_object(mock_search):
    mock_search.return_value = _search_results()

    result = runner.invoke(app, ["search", "what happened"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert isinstance(payload, dict)
    assert "chunks" in payload
    assert "insights" in payload
    assert payload["chunks"][0]["chunk_id"] == "chunk-1"
    assert payload["chunks"][0]["score"] == 0.85
    assert "chunk" in payload["chunks"][0]
    assert "source_id" in payload["chunks"][0]
    assert "source_path" in payload["chunks"][0]
    assert "source_metadata" in payload["chunks"][0]
    mock_search.assert_called_once_with("what happened", limit=10, min_score=0.7)


@patch("rag.cli.hybrid_search")
def test_search_command_passes_limit_and_min_score(mock_search):
    mock_search.return_value = HybridSearchResults(chunks=[], insights=[])

    result = runner.invoke(app, ["search", "what happened", "--limit", "5", "--min-score", "0.3"])

    assert result.exit_code == 0
    mock_search.assert_called_once_with("what happened", limit=5, min_score=0.3)


@patch("rag.cli.hybrid_search")
def test_search_command_short_limit_flag(mock_search):
    mock_search.return_value = HybridSearchResults(chunks=[], insights=[])

    result = runner.invoke(app, ["search", "what happened", "-n", "3"])

    assert result.exit_code == 0
    mock_search.assert_called_once_with("what happened", limit=3, min_score=0.7)


@patch("rag.cli.hybrid_search")
def test_search_command_returns_empty_object_when_no_results(mock_search):
    mock_search.return_value = HybridSearchResults(chunks=[], insights=[])

    result = runner.invoke(app, ["search", "obscure query"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {"chunks": [], "insights": []}
    mock_search.assert_called_once_with("obscure query", limit=10, min_score=0.7)
