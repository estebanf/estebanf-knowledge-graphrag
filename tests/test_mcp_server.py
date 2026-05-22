"""Tests for the MCP server wiring (auth + tool registration)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import tomli_w
from fastapi.testclient import TestClient

from rag.api.auth import KeyStore
from rag.api.main import create_app
from rag.mcp_server import _build_server, build_mcp_app


def test_mcp_server_lists_expected_tools() -> None:
    server = _build_server()
    tool_names = sorted([t.name for t in server._tool_manager.list_tools()])
    assert tool_names == ["community", "list_sources", "retrieve", "search", "source_insights"]


def _make_app_with_keystore(tmp_path: Path):
    keys_file = tmp_path / "keys.toml"
    with keys_file.open("wb") as fh:
        tomli_w.dump({"keys": [{"id": "k", "token": "secret", "scopes": ["read"]}]}, fh)
    from rag.config import settings
    from rag.api import auth as auth_module

    settings.RAG_API_KEYS_PATH = keys_file
    auth_module._default_store = None
    return create_app()


def test_mcp_endpoint_requires_bearer_token(tmp_path: Path) -> None:
    app = _make_app_with_keystore(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/mcp/")
    assert resp.status_code == 401


def test_mcp_endpoint_accepts_valid_bearer(tmp_path: Path) -> None:
    app = _make_app_with_keystore(tmp_path)
    with TestClient(app) as client:
        # Any non-401 response means the auth middleware accepted the token.
        # The MCP transport may then reject the bare GET on its own (e.g., 421
        # for host validation, 406 for missing accept headers) — that's fine.
        resp = client.get("/mcp/", headers={"Authorization": "Bearer secret"})
    assert resp.status_code != 401
