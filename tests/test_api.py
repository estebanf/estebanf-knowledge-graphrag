from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag.api.main import create_app
from rag.retrieval import HybridSearchResults, InsightSearchResult, RetrievalCandidate


def _client() -> TestClient:
    return TestClient(create_app())


def _search_results() -> HybridSearchResults:
    return HybridSearchResults(
        chunks=[
            RetrievalCandidate(
                chunk_id="chunk-1",
                chunk="Search result chunk",
                source_id="source-1",
                source_path="/tmp/source-1.md",
                source_metadata={"kind": "report", "source": "Gartner"},
                score=0.82,
            )
        ],
        insights=[
            InsightSearchResult(
                score=0.91,
                insight="Key market insight",
                insight_id="insight-1",
                topics=["economics"],
                sources=[],
            )
        ],
    )


def _retrieve_results() -> dict:
    return {
        "retrieval_results": [
            {
                "score": 0.91,
                "chunk": "Root chunk",
                "chunk_id": "chunk-1",
                "source_id": "source-1",
                "source_path": "/tmp/source-1.md",
                "source_metadata": {"kind": "report", "source": "Gartner"},
                "related": [
                    {
                        "entity": "Economics",
                        "chunks": [
                            {
                                "score": 0.77,
                                "chunk": "Related chunk",
                                "chunk_id": "chunk-2",
                                "source_id": "source-1",
                                "source_path": "/tmp/source-1.md",
                                "source_metadata": {"kind": "report", "source": "Gartner"},
                            }
                        ],
                        "second_level_related": [],
                    }
                ],
            }
        ]
    }


def _source_detail() -> dict:
    return {
        "source_id": "source-1",
        "name": "Economics of GenAI",
        "file_name": "economics.md",
        "file_type": "text/markdown",
        "storage_path": "/tmp/source-1/original_economics.md",
        "metadata": {"kind": "report", "source": "Gartner"},
        "markdown_content": "# Economics\n\nBody",
    }


def _answer_models() -> list[dict]:
    return [
        {"id": "google/gemma-4-31b-it", "label": "google/gemma-4-31b-it", "default": True},
        {"id": "deepseek/deepseek-v3.2", "label": "deepseek/deepseek-v3.2", "default": False},
    ]


@patch("rag.api.main.get_graph_driver")
@patch("rag.api.main.get_connection")
def test_health_endpoint_reports_ready(mock_conn, mock_graph_driver):
    conn = mock_conn.return_value.__enter__.return_value
    conn.execute.return_value.fetchone.return_value = (1,)
    graph_driver = mock_graph_driver.return_value.__enter__.return_value
    graph_driver.session.return_value.__enter__.return_value.run.return_value = None

    response = _client().get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@patch("rag.api.routes.search.hybrid_search")
def test_search_endpoint_returns_ranked_results(mock_search):
    mock_search.return_value = _search_results()

    response = _client().post("/api/search", json={"query": "economics of agents"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"]["chunks"][0]["chunk_id"] == "chunk-1"
    assert payload["results"]["chunks"][0]["score"] == 0.82
    assert payload["results"]["insights"][0]["insight_id"] == "insight-1"
    assert payload["results"]["insights"][0]["insight"] == "Key market insight"
    mock_search.assert_called_once_with("economics of agents", limit=10, min_score=0.7)


@patch("rag.api.routes.retrieve.retrieve")
def test_retrieve_endpoint_returns_nested_results(mock_retrieve):
    mock_retrieve.return_value = _retrieve_results()

    response = _client().post("/api/retrieve", json={"query": "economics of agents"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_results"][0]["related"][0]["entity"] == "Economics"
    assert payload["retrieval_results"][0]["related"][0]["chunks"][0]["chunk_id"] == "chunk-2"
    mock_retrieve.assert_called_once()


@patch("rag.api.routes.retrieve.retrieve")
def test_retrieve_endpoint_returns_insights(mock_retrieve):
    mock_retrieve.return_value = {
        "retrieval_results": _retrieve_results()["retrieval_results"],
        "insights": [{"insight_id": "i1", "insight": "insight text", "score": 0.88, "topics": ["strategy"]}],
    }

    response = _client().post("/api/retrieve", json={"query": "economics of agents"})

    assert response.status_code == 200
    data = response.json()
    assert "insights" in data
    assert data["insights"][0]["insight_id"] == "i1"


@patch("rag.api.routes.answer.get_supported_answer_models")
def test_answer_models_endpoint_returns_catalog(mock_get_models):
    mock_get_models.return_value = _answer_models()

    response = _client().get("/api/answer/models")

    assert response.status_code == 200
    assert response.json()["models"][0]["id"] == "google/gemma-4-31b-it"
    assert response.json()["models"][0]["default"] is True


@patch("rag.api.routes.answer.stream_answer")
def test_answer_stream_endpoint_streams_answer_then_results(mock_stream_answer):
    mock_stream_answer.return_value = iter([
        'event: answer_delta\ndata: {"delta":"Hello"}\n\n',
        'event: answer_delta\ndata: {"delta":" world"}\n\n',
        'event: results\ndata: {"retrieval_results":[{"chunk_id":"chunk-1"}]}\n\n',
    ])

    with _client().stream(
        "POST",
        "/api/answer/stream",
        json={"query": "economics of agents", "model": "google/gemma-4-31b-it"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert 'event: answer_delta\ndata: {"delta":"Hello"}' in body
    assert 'event: results\ndata: {"retrieval_results":[{"chunk_id":"chunk-1"}]}' in body
    mock_stream_answer.assert_called_once()


@patch("rag.api.routes.sources.get_source_detail")
def test_source_detail_endpoint_returns_markdown(mock_get_source_detail):
    mock_get_source_detail.return_value = _source_detail()

    response = _client().get("/api/sources/source-1")

    assert response.status_code == 200
    assert response.json()["markdown_content"] == "# Economics\n\nBody"
    mock_get_source_detail.assert_called_once_with("source-1")


@patch("rag.api.routes.sources.get_source_detail")
def test_source_detail_endpoint_returns_404_when_missing(mock_get_source_detail):
    mock_get_source_detail.return_value = None

    response = _client().get("/api/sources/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Source not found"


@patch("rag.api.routes.sources.get_source_detail")
def test_source_download_streams_original_file(mock_get_source_detail, tmp_path: Path):
    stored_file = tmp_path / "original_economics.md"
    stored_file.write_text("# Economics\n", encoding="utf-8")
    detail = _source_detail() | {"storage_path": str(stored_file)}
    mock_get_source_detail.return_value = detail

    response = _client().get("/api/sources/source-1/download")

    assert response.status_code == 200
    assert response.content == b"# Economics\n"
    assert "attachment; filename=\"economics.md\"" == response.headers["content-disposition"]


@patch("rag.api.routes.sources.get_source_detail")
def test_source_download_returns_404_when_file_missing(mock_get_source_detail, tmp_path: Path):
    detail = _source_detail() | {"storage_path": str(tmp_path / "missing.md")}
    mock_get_source_detail.return_value = detail

    response = _client().get("/api/sources/source-1/download")

    assert response.status_code == 404
    assert response.json()["detail"] == "Stored file not found"
