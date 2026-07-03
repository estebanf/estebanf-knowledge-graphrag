"""Tests for the backend image-description endpoint (POST /api/prepare/describe-image)."""

from __future__ import annotations

import base64
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag.api.auth import Principal, default_principal_dependency
from rag.api.main import create_app

_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"fakepngbytes").decode()


def _client() -> TestClient:
    return TestClient(create_app())


@patch("rag.api.routes.prepare.describe_image", return_value="A red square.")
def test_describe_image_returns_description(mock_describe) -> None:
    client = _client()
    resp = client.post(
        "/api/prepare/describe-image",
        json={"image_base64": _PNG, "mime_type": "image/png"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"description": "A red square."}
    args, _ = mock_describe.call_args
    assert args[0] == base64.b64decode(_PNG)
    assert args[1] == "image/png"


@patch("rag.api.routes.prepare.describe_image")
def test_describe_image_rejects_unsupported_mime(mock_describe) -> None:
    client = _client()
    resp = client.post(
        "/api/prepare/describe-image",
        json={"image_base64": _PNG, "mime_type": "application/pdf"},
    )
    assert resp.status_code == 415
    mock_describe.assert_not_called()


@patch("rag.api.routes.prepare.describe_image")
def test_describe_image_rejects_malformed_base64(mock_describe) -> None:
    client = _client()
    resp = client.post(
        "/api/prepare/describe-image",
        json={"image_base64": "not!valid!base64!", "mime_type": "image/png"},
    )
    assert resp.status_code == 400
    mock_describe.assert_not_called()


@patch("rag.api.routes.prepare.describe_image")
def test_describe_image_rejects_oversized_payload(mock_describe) -> None:
    client = _client()
    big = base64.b64encode(b"x" * (10 * 1024 * 1024 + 1024)).decode()
    resp = client.post(
        "/api/prepare/describe-image",
        json={"image_base64": big, "mime_type": "image/png"},
    )
    assert resp.status_code == 413
    mock_describe.assert_not_called()


@patch("rag.api.routes.prepare.describe_image", side_effect=RuntimeError("OpenRouter 500"))
def test_describe_image_upstream_error_is_masked(mock_describe) -> None:
    client = _client()
    resp = client.post(
        "/api/prepare/describe-image",
        json={"image_base64": _PNG, "mime_type": "image/png"},
    )
    assert resp.status_code == 502
    assert "OpenRouter" not in resp.text  # upstream detail is not leaked


@patch("rag.api.routes.prepare.describe_image")
def test_describe_image_requires_ingest_scope(mock_describe) -> None:
    # A principal without the ingest scope must be rejected before any LLM call.
    app = create_app()
    app.dependency_overrides[default_principal_dependency] = lambda: Principal(
        kind="apikey", subject="reader", scopes=["read"]
    )
    client = TestClient(app)
    resp = client.post(
        "/api/prepare/describe-image",
        json={"image_base64": _PNG, "mime_type": "image/png"},
    )
    assert resp.status_code == 403
    mock_describe.assert_not_called()
