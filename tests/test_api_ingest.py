"""Tests for the ingest API: multipart upload + JSON text submission."""

from __future__ import annotations

import io
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag.api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


@patch("rag.api.routes.ingest.submit_ingestion_job")
def test_ingest_multipart_invokes_submit_with_temp_file(mock_submit, tmp_path) -> None:
    captured: dict = {}

    def fake_submit(path, name=None, metadata=None):
        captured["name"] = path.name
        captured["bytes"] = path.read_bytes()
        captured["kw_name"] = name
        captured["kw_metadata"] = metadata
        return {"source_id": "src-1", "job_id": "job-1", "status": "pending"}

    mock_submit.side_effect = fake_submit
    client = _client()
    files = {"file": ("hello.md", b"# hi\n\ncontent", "text/markdown")}
    resp = client.post(
        "/api/ingest",
        files=files,
        data={"name": "Hello", "metadata": '{"kind":"note"}'},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"source_id": "src-1", "job_id": "job-1", "status": "pending"}
    assert captured["name"] == "hello.md"
    assert captured["bytes"] == b"# hi\n\ncontent"
    assert captured["kw_name"] == "Hello"
    assert captured["kw_metadata"] == {"kind": "note"}


@patch("rag.api.routes.ingest.submit_ingestion_job")
def test_ingest_multipart_without_metadata(mock_submit) -> None:
    mock_submit.return_value = {"source_id": "s", "job_id": "j", "status": "pending"}
    client = _client()
    resp = client.post("/api/ingest", files={"file": ("a.md", b"x", "text/markdown")})
    assert resp.status_code == 200
    args, kwargs = mock_submit.call_args
    assert kwargs == {"name": None, "metadata": None}


def test_ingest_multipart_rejects_unsupported_extension() -> None:
    client = _client()
    resp = client.post("/api/ingest", files={"file": ("x.exe", b"x", "application/octet-stream")})
    assert resp.status_code == 415


@patch("rag.api.routes.ingest.submit_ingestion_job")
def test_ingest_propagates_duplicate_error(mock_submit) -> None:
    mock_submit.side_effect = ValueError("Duplicate: file already ingested as source abc")
    client = _client()
    resp = client.post("/api/ingest", files={"file": ("a.md", b"x", "text/markdown")})
    assert resp.status_code == 409
    assert "Duplicate" in resp.json()["detail"]


@patch("rag.api.routes.ingest.submit_ingestion_job")
def test_ingest_text_endpoint(mock_submit) -> None:
    captured: dict = {}

    def fake_submit(path, name=None, metadata=None):
        captured["suffix"] = path.suffix
        captured["text"] = path.read_text()
        captured["kw"] = {"name": name, "metadata": metadata}
        return {"source_id": "s", "job_id": "j", "status": "pending"}

    mock_submit.side_effect = fake_submit
    client = _client()
    resp = client.post(
        "/api/ingest/text",
        json={"content": "# Hello\nbody", "name": "hi", "metadata": {"k": "v"}},
    )
    assert resp.status_code == 200
    assert captured["suffix"] == ".md"
    assert captured["text"] == "# Hello\nbody"
    assert captured["kw"] == {"name": "hi", "metadata": {"k": "v"}}
