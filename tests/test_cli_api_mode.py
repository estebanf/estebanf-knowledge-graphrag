"""Exercise the API-backed code paths in the CLI via a mocked RagClient."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from rag import cli as cli_module


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def api_env(monkeypatch):
    monkeypatch.setenv("RAG_SERVER_URL", "http://test")
    monkeypatch.setenv("RAG_API_KEY", "k")


@pytest.fixture
def fake_client(monkeypatch):
    """Make ``_get_client()`` return a MagicMock with context-manager semantics."""
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    monkeypatch.setattr(cli_module, "_get_client", lambda: client)
    return client


def test_health_uses_api(runner: CliRunner, api_env, fake_client) -> None:
    fake_client.health.return_value = {"status": "ready"}
    result = runner.invoke(cli_module.app, ["health"])
    assert result.exit_code == 0, result.output
    fake_client.health.assert_called_once()
    assert "Ready: server" in result.output


def test_search_uses_api(runner: CliRunner, api_env, fake_client) -> None:
    fake_client.search.return_value = {"results": {"chunks": [], "insights": []}}
    result = runner.invoke(cli_module.app, ["search", "what", "--limit", "3", "--min-score", "0.4"])
    assert result.exit_code == 0
    fake_client.search.assert_called_once_with("what", limit=3, min_score=0.4)


def test_jobs_list_uses_api(runner: CliRunner, api_env, fake_client) -> None:
    fake_client.list_jobs.return_value = {"jobs": [{"id": "j1", "source_id": "s1", "status": "completed", "current_stage": None, "stage_log": {}, "created_at": "2026-05-01T12:00:00", "updated_at": "2026-05-01T12:30:00"}]}
    result = runner.invoke(cli_module.app, ["jobs", "list"])
    assert result.exit_code == 0
    fake_client.list_jobs.assert_called_once_with(status=None)
    assert "j1" in result.output


def test_worker_launch_uses_api(runner: CliRunner, api_env, fake_client) -> None:
    fake_client.launch_workers.return_value = {"ids": ["w1", "w2"]}
    result = runner.invoke(cli_module.app, ["worker", "launch", "2"])
    assert result.exit_code == 0
    fake_client.launch_workers.assert_called_once_with(2)
    assert "w1" in result.output and "w2" in result.output


def test_worker_stop_uses_api(runner: CliRunner, api_env, fake_client) -> None:
    fake_client.stop_worker.return_value = {"worker_id": "w1", "status": "stopped"}
    result = runner.invoke(cli_module.app, ["worker", "stop", "w1"])
    assert result.exit_code == 0
    fake_client.stop_worker.assert_called_once_with("w1")


def test_worker_stop_all_uses_api(runner: CliRunner, api_env, fake_client) -> None:
    fake_client.stop_all_workers.return_value = {"stopped": ["w1", "w2"]}
    result = runner.invoke(cli_module.app, ["worker", "stop", "--all"])
    assert result.exit_code == 0
    fake_client.stop_all_workers.assert_called_once()
    assert "w1" in result.output and "w2" in result.output


def test_worker_stop_requires_id_or_all(runner: CliRunner, api_env, fake_client) -> None:
    result = runner.invoke(cli_module.app, ["worker", "stop"])
    assert result.exit_code != 0


def test_worker_list_excludes_stopped_by_default(runner: CliRunner, api_env, fake_client) -> None:
    fake_client.list_workers.return_value = {"workers": []}
    runner.invoke(cli_module.app, ["worker", "list"])
    fake_client.list_workers.assert_called_with(include_stopped=False)

    fake_client.list_workers.reset_mock()
    runner.invoke(cli_module.app, ["worker", "list", "--all"])
    fake_client.list_workers.assert_called_with(include_stopped=True)


def test_worker_list_uses_api(runner: CliRunner, api_env, fake_client) -> None:
    fake_client.list_workers.return_value = {"workers": [
        {"id": "w1", "pid": 1, "status": "running", "started_at": 1700000000.0, "stopped_at": None, "exit_code": None, "log_path": "/x", "host": "h"},
    ]}
    result = runner.invoke(cli_module.app, ["worker", "list"])
    assert result.exit_code == 0, result.output
    fake_client.list_workers.assert_called_once()
    assert "w1" in result.output


def test_ingest_uses_api(runner: CliRunner, api_env, fake_client, tmp_path) -> None:
    fake_client.submit_ingest.return_value = {"job_id": "j1", "status": "pending", "source_id": "s1"}
    file = tmp_path / "a.md"
    file.write_text("hello")
    result = runner.invoke(cli_module.app, ["ingest", str(file)])
    assert result.exit_code == 0, result.output
    fake_client.submit_ingest.assert_called_once()
    assert "j1" in result.output


def test_configure_writes_config(runner: CliRunner, monkeypatch, tmp_path) -> None:
    from rag import cli_config

    monkeypatch.setattr(cli_config, "DEFAULT_PATH", tmp_path / "c.toml")
    result = runner.invoke(cli_module.app, ["configure", "--server-url", "http://x", "--api-key", "y"])
    assert result.exit_code == 0
    saved = (tmp_path / "c.toml").read_text()
    assert "http://x" in saved and "y" in saved
