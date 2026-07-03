"""Tests for the RagClient HTTP wrapper using httpx.MockTransport."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from rag.api_client import ApiError, RagClient


def _mock(handler) -> RagClient:
    transport = httpx.MockTransport(handler)
    return RagClient.with_transport(transport)


def test_health_returns_payload() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path == "/api/health"
        assert req.headers["authorization"] == "Bearer test"
        return httpx.Response(200, json={"status": "ready"})

    with _mock(handler) as client:
        assert client.health() == {"status": "ready"}


def test_search_posts_body() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"results": {"chunks": [], "insights": []}})

    with _mock(handler) as client:
        result = client.search("hello", limit=3, min_score=0.5)
    assert seen["body"] == {"query": "hello", "limit": 3, "min_score": 0.5}
    assert result == {"results": {"chunks": [], "insights": []}}


def test_retrieve_omits_none_values() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={})

    with _mock(handler) as client:
        client.retrieve("q", source_ids=["s1"], rrf_k=None, result_count=2)
    assert seen["body"] == {"query": "q", "source_ids": ["s1"], "result_count": 2}


def test_list_sources_serializes_filters() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"sources": [], "total": 0, "limit": 5, "offset": 0})

    with _mock(handler) as client:
        client.list_sources(limit=5, offset=0, metadata=["kind:report", "x:y"], q="hello")
    assert "limit=5" in seen["url"]
    assert "metadata=kind%3Areport" in seen["url"]
    assert "metadata=x%3Ay" in seen["url"]
    assert "q=hello" in seen["url"]


def test_delete_source_sends_hard_flag() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"source_id": "x", "hard": True})

    with _mock(handler) as client:
        client.delete_source("x", hard=True)
    assert seen["method"] == "DELETE"
    assert "hard=true" in seen["url"]


def test_submit_ingest_sends_multipart(tmp_path: Path) -> None:
    file = tmp_path / "test.md"
    file.write_bytes(b"# hi")
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["content_type"] = req.headers.get("content-type", "")
        seen["body"] = req.content
        return httpx.Response(200, json={"source_id": "s", "job_id": "j", "status": "pending"})

    with _mock(handler) as client:
        result = client.submit_ingest(file, name="hi", metadata={"k": "v"})
    assert "multipart/form-data" in seen["content_type"]
    assert b"# hi" in seen["body"]
    assert b'name="name"' in seen["body"]
    assert result == {"source_id": "s", "job_id": "j", "status": "pending"}


def test_describe_image_posts_base64() -> None:
    import base64

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"description": "A chart."})

    with _mock(handler) as client:
        result = client.describe_image(b"rawbytes", "image/png")
    assert result == "A chart."
    assert seen["path"] == "/api/prepare/describe-image"
    assert seen["body"] == {
        "image_base64": base64.b64encode(b"rawbytes").decode(),
        "mime_type": "image/png",
    }


def test_submit_text_preserves_original_provenance() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"source_id": "s", "job_id": "j", "status": "pending"})

    with _mock(handler) as client:
        result = client.submit_text(
            "# prepared\n\ncontent",
            name="Report",
            metadata={"k": "v"},
            original_md5="abc123",
            file_name="report.pdf",
            file_type="pdf",
        )
    assert seen["path"] == "/api/ingest/text"
    assert seen["body"] == {
        "content": "# prepared\n\ncontent",
        "name": "Report",
        "metadata": {"k": "v"},
        "original_md5": "abc123",
        "file_name": "report.pdf",
        "file_type": "pdf",
    }
    assert result == {"source_id": "s", "job_id": "j", "status": "pending"}


def test_submit_text_omits_none_fields() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"source_id": "s", "job_id": "j", "status": "pending"})

    with _mock(handler) as client:
        client.submit_text("# md")
    assert seen["body"] == {"content": "# md"}


def test_submit_text_raises_on_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "Duplicate: already ingested"})

    with _mock(handler) as client:
        with pytest.raises(ApiError) as info:
            client.submit_text("# md", original_md5="dup")
    assert info.value.status == 409
    assert "Duplicate" in info.value.detail


def test_list_jobs_status_param() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"jobs": []})

    with _mock(handler) as client:
        client.list_jobs(status="failed")
    assert "status=failed" in seen["url"]


def test_retry_job_includes_from_stage() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"job_id": "x", "retry_from_stage": "chunking"})

    with _mock(handler) as client:
        client.retry_job("x", from_stage="chunking")
    assert "/api/jobs/x/retry" in seen["url"]
    assert seen["body"] == {"from_stage": "chunking"}


def test_launch_and_stop_workers() -> None:
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(f"{req.method} {req.url.path}?{req.url.query.decode()}")
        if "stop" in req.url.path:
            return httpx.Response(200, json={"worker_id": "w1", "status": "stopped"})
        return httpx.Response(200, json={"ids": ["w1", "w2"]})

    with _mock(handler) as client:
        result = client.launch_workers(2)
        client.stop_worker("w1")
    assert result == {"ids": ["w1", "w2"]}
    assert any("n=2" in c for c in calls)


def test_error_response_raises_api_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "Duplicate"})

    with _mock(handler) as client:
        with pytest.raises(ApiError) as info:
            client.search("q", limit=1, min_score=0.1)
    assert info.value.status == 409
    assert "Duplicate" in info.value.detail
