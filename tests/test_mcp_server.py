"""Tests for the MCP server wiring (auth + tool registration)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tomli_w
from fastapi.testclient import TestClient

from rag.api.auth import KeyStore
from rag.api.main import create_app
from rag.mcp_server import _build_server, build_mcp_app


def test_mcp_server_lists_expected_tools() -> None:
    server = _build_server()
    tool_names = sorted([t.name for t in server._tool_manager.list_tools()])
    assert tool_names == [
        "community",
        "get_answer",
        "get_community_run",
        "get_theme_report",
        "get_working_set",
        "list_answers",
        "list_community_runs",
        "list_metadata_facets",
        "list_sources",
        "list_theme_reports",
        "list_working_sets",
        "retrieve",
        "search",
        "source_insights",
    ]


def _find_tool(server, name: str):
    for t in server._tool_manager.list_tools():
        if t.name == name:
            return t
    return None


def test_community_tool_passes_resolution_and_source_cooc_weight() -> None:
    server = _build_server()
    tool = _find_tool(server, "community")
    assert tool is not None
    fn = tool.fn
    with patch("rag.community.detect_communities") as mock_detect:
        mock_detect.return_value = {"metadata": {}, "communities": []}
        fn(
            scope_mode="ids",
            source_ids=["s1"],
            community_options={"resolution": 1.5, "source_cooc_weight": 0.2, "cross_source_top_k": 10, "max_cross_source_queries": 50},
        )
        call_kwargs = mock_detect.call_args.kwargs
        assert call_kwargs["resolution"] == 1.5
        assert call_kwargs["source_cooc_weight"] == 0.2
        assert call_kwargs["cross_source_top_k"] == 10
        assert call_kwargs["max_cross_source_queries"] == 50


def test_community_tool_working_set_scope_resolves_source_ids() -> None:
    server = _build_server()
    tool = _find_tool(server, "community")
    assert tool is not None
    fn = tool.fn

    import json

    mock_row = MagicMock()
    mock_row.__getitem__ = MagicMock(return_value=json.dumps(["s_a", "s_b"]))
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.execute.return_value.fetchone.return_value = mock_row

    with patch("rag.community.detect_communities") as mock_detect, \
         patch("rag.db.get_connection", return_value=mock_conn):
        mock_detect.return_value = {"metadata": {}, "communities": []}
        fn(scope_mode="working_set", working_set_id="ws1")
        call_kwargs = mock_detect.call_args.kwargs
        assert call_kwargs["scope_mode"] == "ids"
        assert call_kwargs["source_ids"] == ["s_a", "s_b"]


def test_community_tool_working_set_missing_id_raises() -> None:
    server = _build_server()
    tool = _find_tool(server, "community")
    assert tool is not None
    fn = tool.fn
    with pytest.raises(ValueError, match="working_set_id is required"):
        fn(scope_mode="working_set")


def test_community_tool_working_set_not_found_raises() -> None:
    server = _build_server()
    tool = _find_tool(server, "community")
    assert tool is not None
    fn = tool.fn

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.execute.return_value.fetchone.return_value = None

    with patch("rag.db.get_connection", return_value=mock_conn):
        with pytest.raises(ValueError, match="working set not found"):
            fn(scope_mode="working_set", working_set_id="ws_x")


def test_get_community_run_returns_result() -> None:
    server = _build_server()
    tool = _find_tool(server, "get_community_run")
    assert tool is not None
    fn = tool.fn

    expected = {"id": "r1", "status": "completed", "params": {}, "source_ids": [], "stage_log": [], "result": None, "error": None, "created_at": None, "updated_at": None}
    with patch("rag.community_runs.get_run", return_value=expected) as mock_get:
        result = fn(run_id="r1")
    mock_get.assert_called_once_with("r1")
    assert result == expected


def test_get_community_run_not_found_raises() -> None:
    server = _build_server()
    tool = _find_tool(server, "get_community_run")
    assert tool is not None
    fn = tool.fn

    with patch("rag.community_runs.get_run", return_value=None):
        with pytest.raises(ValueError, match="community run not found"):
            fn(run_id="r_x")


def test_get_theme_report_returns_result() -> None:
    server = _build_server()
    tool = _find_tool(server, "get_theme_report")
    assert tool is not None
    fn = tool.fn

    expected = {"id": "t1", "run_id": "r1", "status": "completed", "failed_community_ids": [], "report": {}, "model": "", "created_at": None}
    with patch("rag.themes.get_report", return_value=expected) as mock_get:
        result = fn(report_id="t1")
    mock_get.assert_called_once_with("t1")
    assert result == expected


def test_get_theme_report_not_found_raises() -> None:
    server = _build_server()
    tool = _find_tool(server, "get_theme_report")
    assert tool is not None
    fn = tool.fn

    with patch("rag.themes.get_report", return_value=None):
        with pytest.raises(ValueError, match="theme report not found"):
            fn(report_id="t_x")


def test_list_community_runs_calls_list_runs() -> None:
    server = _build_server()
    tool = _find_tool(server, "list_community_runs")
    assert tool is not None
    fn = tool.fn

    expected = {"runs": [], "total": 0, "limit": 5, "offset": 0}
    with patch("rag.community_runs.list_runs", return_value=expected) as mock_list:
        result = fn(limit=5, offset=0)
    mock_list.assert_called_once_with(limit=5, offset=0)
    assert result == expected


def test_list_theme_reports_calls_list_reports() -> None:
    server = _build_server()
    tool = _find_tool(server, "list_theme_reports")
    assert tool is not None
    fn = tool.fn

    expected = {"reports": [], "total": 0, "limit": 5, "offset": 0}
    with patch("rag.themes.list_reports", return_value=expected) as mock_list:
        result = fn(limit=5, offset=0)
    mock_list.assert_called_once_with(limit=5, offset=0)
    assert result == expected


def test_tool_listing_contains_no_write_tool() -> None:
    server = _build_server()
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    write_patterns = ["generate", "create", "save", "update", "delete", "write", "regenerate", "start_run", "start_community"]
    for name in tool_names:
        for pattern in write_patterns:
            assert pattern not in name, f"Write/generate tool detected: {name}"


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
