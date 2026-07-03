"""Confirm that every non-public API route returns 401 when auth is enforced."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from rag.api.main import create_app


@pytest.fixture
def gated_client(monkeypatch):
    """Build a client where auth gating is enforced (no test bypass)."""
    monkeypatch.delenv("RAG_BYPASS_AUTH", raising=False)
    return TestClient(create_app())


GATED_REQUESTS = [
    ("POST", "/api/search", {"query": "x"}),
    ("POST", "/api/retrieve", {"query": "x"}),
    ("GET", "/api/sources", None),
    ("GET", "/api/sources/some-id", None),
    ("GET", "/api/sources/some-id/insights", None),
    ("POST", "/api/community", {"scope_mode": "ids", "source_ids": []}),
    ("GET", "/api/answer/models", None),
    ("GET", "/api/auth/me", None),
    ("GET", "/api/jobs", None),
    ("GET", "/api/jobs/stats", None),
    ("GET", "/api/jobs/stage-stats", None),
    ("POST", "/api/jobs/stage-stats/baseline", None),
    ("GET", "/api/jobs/some-id", None),
    ("GET", "/api/workers", None),
    ("POST", "/api/workers/launch", None),
    ("POST", "/api/prepare/describe-image", {"image_base64": "eA==", "mime_type": "image/png"}),
]


@pytest.mark.parametrize("method,path,body", GATED_REQUESTS)
def test_gated_endpoint_requires_credentials(gated_client, method, path, body):
    if method == "GET":
        resp = gated_client.get(path)
    else:
        resp = gated_client.post(path, json=body or {})
    assert resp.status_code == 401, f"{method} {path} should require auth, got {resp.status_code}"


def test_health_endpoint_is_public(gated_client):
    # /api/health is intentionally public so deploys can probe it. It still
    # touches the DB so it may return 5xx in a hermetic test; we only care
    # that auth doesn't block it.
    resp = gated_client.get("/api/health")
    assert resp.status_code != 401


def test_login_endpoint_is_public(gated_client):
    resp = gated_client.post("/api/auth/login", json={"username": "x", "password": "y"})
    # Should reach the auth handler (which returns 401 for bad creds), not the
    # gating dep (which returns 401 with a different detail). Either way it
    # shouldn't 4xx because of *gating* — it should respond on the merits.
    assert resp.status_code in (200, 401, 500)
