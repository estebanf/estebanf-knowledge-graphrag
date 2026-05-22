"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Bypass the API gating dependency for the existing test suite.
# `create_app()` reads this env var on every invocation, so tests created
# anywhere in the suite automatically get a no-op auth dep that returns a
# fake `Principal` with full scopes. Tests that exercise auth itself build
# their own ``FastAPI`` instances without using ``create_app()``.
os.environ.setdefault("RAG_BYPASS_AUTH", "1")


@pytest.fixture(autouse=True)
def _isolate_cli_config(monkeypatch, tmp_path):
    """Point the CLI config at a non-existent temp path and clear env overrides.

    Without this, any RAG_SERVER_URL/RAG_API_KEY in the developer's shell or any
    pre-existing ``~/.config/rag/config.toml`` would silently flip CLI tests
    into API mode and break the assertions that target the local-DB code path.
    """
    monkeypatch.delenv("RAG_SERVER_URL", raising=False)
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    from rag import cli_config

    monkeypatch.setattr(cli_config, "DEFAULT_PATH", tmp_path / "no-config.toml")
    yield
