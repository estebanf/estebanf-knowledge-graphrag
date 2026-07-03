"""HTTP client used by the CLI to talk to the RAG API server."""

from __future__ import annotations

import base64
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx
from httpx_sse import connect_sse

from rag.cli_config import CliConfig, require_config


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"API error {status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass
class RagClient:
    base_url: str
    api_key: str
    timeout: float = 360.0
    _client: Optional[httpx.Client] = None
    _owns_client: bool = True

    @classmethod
    def from_config(cls, cfg: Optional[CliConfig] = None) -> "RagClient":
        cfg = cfg or require_config()
        return cls(base_url=cfg.server_url.rstrip("/"), api_key=cfg.api_key)

    def __post_init__(self) -> None:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )

    @classmethod
    def with_transport(cls, transport: httpx.BaseTransport, base_url: str = "http://test", api_key: str = "test") -> "RagClient":
        client = httpx.Client(
            transport=transport,
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return cls(base_url=base_url, api_key=api_key, _client=client, _owns_client=False)

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()

    def __enter__(self) -> "RagClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- low-level helpers -----------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        resp = self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise ApiError(resp.status_code, str(detail))
        return resp

    def _get_json(self, path: str, **kwargs) -> Any:
        return self._request("GET", path, **kwargs).json()

    def _post_json(self, path: str, payload: dict | None = None, **kwargs) -> Any:
        return self._request("POST", path, json=payload, **kwargs).json()

    # --- health ----------------------------------------------------------------

    def health(self) -> dict:
        return self._get_json("/api/health")

    # --- auth ------------------------------------------------------------------

    def me(self) -> dict:
        return self._get_json("/api/auth/me")

    # --- search / retrieve / community / answer --------------------------------

    def search(self, query: str, *, limit: int, min_score: float) -> dict:
        return self._post_json("/api/search", {"query": query, "limit": limit, "min_score": min_score})

    def retrieve(self, query: str, **opts: Any) -> dict:
        body = {"query": query, **{k: v for k, v in opts.items() if v is not None}}
        return self._post_json("/api/retrieve", body)

    def community(self, payload: dict) -> dict:
        return self._post_json("/api/community", payload)

    def answer_models(self) -> dict:
        return self._get_json("/api/answer/models")

    def stream_answer(self, query: str, model: str) -> Iterator[dict]:
        """Yield SSE events with ``event``/``data`` parsed."""
        with connect_sse(
            self._client,
            "POST",
            "/api/answer/stream",
            json={"query": query, "model": model},
        ) as event_source:
            for sse in event_source.iter_sse():
                try:
                    data = json.loads(sse.data) if sse.data else {}
                except json.JSONDecodeError:
                    data = {"raw": sse.data}
                yield {"event": sse.event, "data": data}

    # --- sources ---------------------------------------------------------------

    def list_sources(self, *, limit: int = 20, offset: int = 0, metadata: list[str] | None = None, q: str | None = None) -> dict:
        params: list[tuple[str, str]] = [("limit", str(limit)), ("offset", str(offset))]
        for item in metadata or []:
            params.append(("metadata", item))
        if q:
            params.append(("q", q))
        return self._get_json("/api/sources", params=params)

    def get_source(self, source_id: str) -> dict:
        return self._get_json(f"/api/sources/{source_id}")

    def source_insights(self, source_id: str) -> dict:
        return self._get_json(f"/api/sources/{source_id}/insights")

    def delete_source(self, source_id: str, *, hard: bool = False) -> dict:
        return self._request("DELETE", f"/api/sources/{source_id}", params={"hard": "true" if hard else "false"}).json()

    # --- ingest ----------------------------------------------------------------

    def submit_ingest(self, file_path: Path, *, name: str | None = None, metadata: dict | None = None) -> dict:
        with open(file_path, "rb") as fh:
            files = {"file": (file_path.name, fh, "application/octet-stream")}
            data: dict[str, str] = {}
            if name is not None:
                data["name"] = name
            if metadata is not None:
                data["metadata"] = json.dumps(metadata)
            resp = self._client.post("/api/ingest", files=files, data=data, timeout=None)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise ApiError(resp.status_code, str(detail))
        return resp.json()

    def describe_image(self, image_bytes: bytes, mime_type: str) -> str:
        """Ask the backend to describe an extracted image (prepared ingestion).

        Keeps LLM credentials on the server: the CLI sends transient base64 image
        bytes and receives a text description.
        """
        b64 = base64.b64encode(image_bytes).decode()
        payload = self._post_json(
            "/api/prepare/describe-image",
            {"image_base64": b64, "mime_type": mime_type},
        )
        return payload["description"]

    def submit_text(
        self,
        content: str,
        *,
        name: str | None = None,
        metadata: dict | None = None,
        original_md5: str | None = None,
        file_name: str | None = None,
        file_type: str | None = None,
    ) -> dict:
        """Submit prepared markdown/text as an ingestion job.

        ``original_md5``/``file_name``/``file_type`` carry original-source
        provenance for prepared binary documents so duplicate detection keys on
        the original binary hash and the source row records what the operator
        actually ingested (R9, R10, R11).
        """
        payload: dict = {"content": content}
        if name is not None:
            payload["name"] = name
        if metadata is not None:
            payload["metadata"] = metadata
        if original_md5 is not None:
            payload["original_md5"] = original_md5
        if file_name is not None:
            payload["file_name"] = file_name
        if file_type is not None:
            payload["file_type"] = file_type
        return self._post_json("/api/ingest/text", payload)

    # --- jobs ------------------------------------------------------------------

    def list_jobs(self, *, status: str | None = None) -> dict:
        params = {"status": status} if status else None
        return self._get_json("/api/jobs", params=params)

    def job_stats(self) -> dict:
        return self._get_json("/api/jobs/stats")

    def get_job(self, job_id: str) -> dict:
        return self._get_json(f"/api/jobs/{job_id}")

    def retry_job(self, job_id: str, *, from_stage: str | None = None) -> dict:
        return self._post_json(f"/api/jobs/{job_id}/retry", {"from_stage": from_stage})

    def cancel_job(self, job_id: str) -> dict:
        return self._post_json(f"/api/jobs/{job_id}/cancel")

    # --- workers ---------------------------------------------------------------

    def launch_workers(self, n: int = 1) -> dict:
        return self._post_json("/api/workers/launch", params={"n": n}) if False else self._request(
            "POST", "/api/workers/launch", params={"n": n}
        ).json()

    def stop_worker(self, worker_id: str) -> dict:
        return self._request("POST", f"/api/workers/{worker_id}/stop").json()

    def stop_all_workers(self) -> dict:
        return self._request("POST", "/api/workers/stop-all").json()

    def list_workers(self, *, include_stopped: bool = False) -> dict:
        params = {"all": "true"} if include_stopped else None
        return self._get_json("/api/workers", params=params)

    def worker_log(self, worker_id: str) -> str:
        return self._request("GET", f"/api/workers/{worker_id}/log").text

    def follow_worker_log(self, worker_id: str) -> Iterator[str]:
        with connect_sse(self._client, "GET", f"/api/workers/{worker_id}/log", params={"follow": "true"}) as event_source:
            for sse in event_source.iter_sse():
                yield sse.data
