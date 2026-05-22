"""Tests for ``rag.cli_config``: server URL + API key resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

import rag.cli_config as cli_config
from rag.cli_config import CliConfig, load_cli_config, save_cli_config


def test_env_vars_take_precedence(monkeypatch, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('server_url = "http://from-file:8000"\napi_key = "filekey"\n')
    monkeypatch.setenv("RAG_SERVER_URL", "http://from-env:9000")
    monkeypatch.setenv("RAG_API_KEY", "envkey")

    cfg = load_cli_config(path=path)
    assert cfg.server_url == "http://from-env:9000"
    assert cfg.api_key == "envkey"


def test_file_used_when_env_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_SERVER_URL", raising=False)
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    path = tmp_path / "config.toml"
    path.write_text('server_url = "http://srv:8000"\napi_key = "kkk"\n')

    cfg = load_cli_config(path=path)
    assert cfg == CliConfig(server_url="http://srv:8000", api_key="kkk")


def test_missing_file_yields_empty_config(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_SERVER_URL", raising=False)
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    cfg = load_cli_config(path=tmp_path / "missing.toml")
    assert cfg.server_url is None
    assert cfg.api_key is None


def test_save_and_reload_round_trip(tmp_path):
    path = tmp_path / "config.toml"
    save_cli_config(CliConfig(server_url="http://x", api_key="abc"), path=path)
    text = path.read_text()
    assert "http://x" in text and "abc" in text
    loaded = load_cli_config(path=path)
    assert loaded.server_url == "http://x"
    assert loaded.api_key == "abc"


def test_require_returns_validated_config(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_SERVER_URL", "http://x")
    monkeypatch.setenv("RAG_API_KEY", "y")
    cfg = cli_config.require_config(path=tmp_path / "x.toml")
    assert cfg.server_url == "http://x"


def test_require_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_SERVER_URL", raising=False)
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    with pytest.raises(cli_config.CliConfigError):
        cli_config.require_config(path=tmp_path / "missing.toml")
