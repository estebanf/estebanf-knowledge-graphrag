import json
from unittest.mock import patch

from typer.testing import CliRunner

from rag.cli import app


runner = CliRunner()


def _retrieval_response() -> dict:
    return {
        "retrieval_results": [
            {
                "score": 0.91,
                "chunk": "Root chunk",
                "chunk_id": "chunk-1",
                "source_id": "source-1",
                "source_path": "/tmp/source-1.md",
                "source_metadata": {"kind": "report"},
                "related": [],
            }
        ]
    }


@patch("rag.cli.retrieve")
def test_retrieve_command_prints_json_response(mock_retrieve):
    mock_retrieve.return_value = _retrieval_response()

    result = runner.invoke(app, ["retrieve", "what happened"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["retrieval_results"][0]["chunk_id"] == "chunk-1"
    mock_retrieve.assert_called_once()


@patch("rag.cli.retrieve")
def test_retrieve_command_passes_filters_and_overrides(mock_retrieve):
    mock_retrieve.return_value = _retrieval_response()

    result = runner.invoke(
        app,
        [
            "retrieve",
            "what happened",
            "--source-id",
            "source-1",
            "--source-id",
            "source-2",
            "--filter",
            "kind=report",
            "--filter",
            "domain=technical",
            "--seed-count",
            "7",
            "--result-count",
            "3",
            "--rrf-k",
            "40",
            "--entity-confidence-threshold",
            "0.8",
            "--first-hop-similarity-threshold",
            "0.6",
            "--second-hop-similarity-threshold",
            "0.7",
        ],
    )

    assert result.exit_code == 0
    mock_retrieve.assert_called_once_with(
        query="what happened",
        source_ids=["source-1", "source-2"],
        filters={"kind": "report", "domain": "technical"},
        seed_count=7,
        result_count=3,
        rrf_k=40,
        entity_confidence_threshold=0.8,
        first_hop_similarity_threshold=0.6,
        second_hop_similarity_threshold=0.7,
        trace=False,
        trace_printer=None,
    )


@patch("rag.cli.retrieve")
def test_retrieve_command_trace_prints_activity_before_json(mock_retrieve):
    def _side_effect(**kwargs):
        kwargs["trace_printer"]("generated variants")
        kwargs["trace_printer"]("reranked seeds")
        return _retrieval_response()

    mock_retrieve.side_effect = _side_effect

    result = runner.invoke(app, ["retrieve", "what happened", "--trace"])

    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert lines[0] == "[trace] generated variants"
    assert lines[1] == "[trace] reranked seeds"
    payload = json.loads("\n".join(lines[2:]))
    assert payload["retrieval_results"][0]["chunk_id"] == "chunk-1"
    assert mock_retrieve.call_args.kwargs["trace"] is True
    assert callable(mock_retrieve.call_args.kwargs["trace_printer"])


@patch("rag.cli.retrieve")
def test_retrieve_command_rejects_invalid_filter_format(mock_retrieve):
    result = runner.invoke(app, ["retrieve", "what happened", "--filter", "kind"])

    assert result.exit_code == 1
    assert "Invalid filter format" in result.output
    mock_retrieve.assert_not_called()
