"""Tests for the explicit `rag prepare` command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from rag import cli as cli_module
from rag.prepare import ExtractedImage, PreparedDocument

runner = CliRunner()


@pytest.fixture
def api_env(monkeypatch):
    monkeypatch.setenv("RAG_SERVER_URL", "http://test")
    monkeypatch.setenv("RAG_API_KEY", "k")


@pytest.fixture
def fake_client(monkeypatch):
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    monkeypatch.setattr(cli_module, "_get_client", lambda: client)
    return client


def _prepared(images):
    return PreparedDocument(
        markdown="# deck\n\n<!-- image -->" if images else "# deck\n\ntext only",
        images=images,
        original_filename="deck.pptx",
        original_extension="pptx",
        original_md5="hash",
        image_count=len(images),
    )


def test_prepare_writes_markdown_with_backend_descriptions(api_env, fake_client, tmp_path):
    prepared = _prepared([ExtractedImage(data=b"img", mime_type="image/png")])
    out = tmp_path / "prepared.md"
    src = tmp_path / "deck.pptx"
    src.write_bytes(b"pptx")

    with patch.object(cli_module, "prepare_binary", return_value=prepared), patch.object(
        cli_module, "finalize_markdown", return_value="# deck\n\nA workflow diagram."
    ):
        result = runner.invoke(cli_module.app, ["prepare", str(src), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert out.read_text() == "# deck\n\nA workflow diagram."


def test_prepare_overwrites_existing_out_file(api_env, fake_client, tmp_path):
    prepared = _prepared([])
    out = tmp_path / "prepared.md"
    out.write_text("STALE", encoding="utf-8")
    src = tmp_path / "deck.pptx"
    src.write_bytes(b"pptx")

    with patch.object(cli_module, "prepare_binary", return_value=prepared):
        result = runner.invoke(cli_module.app, ["prepare", str(src), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert "STALE" not in out.read_text()
    assert "text only" in out.read_text()


def test_prepare_without_api_config_and_images_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_SERVER_URL", raising=False)
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    prepared = _prepared([ExtractedImage(data=b"img", mime_type="image/png")])
    out = tmp_path / "prepared.md"
    src = tmp_path / "deck.pptx"
    src.write_bytes(b"pptx")

    with patch.object(cli_module, "prepare_binary", return_value=prepared), patch.object(
        cli_module, "describe_image"
    ) as local_describe:
        result = runner.invoke(cli_module.app, ["prepare", str(src), "--out", str(out)])

    assert result.exit_code != 0
    assert "configure" in result.output.lower()
    local_describe.assert_not_called()
    assert not out.exists()


def test_prepare_no_images_needs_no_backend(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_SERVER_URL", raising=False)
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    prepared = _prepared([])
    out = tmp_path / "prepared.md"
    src = tmp_path / "deck.pptx"
    src.write_bytes(b"pptx")

    with patch.object(cli_module, "prepare_binary", return_value=prepared):
        result = runner.invoke(cli_module.app, ["prepare", str(src), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert "text only" in out.read_text()


def test_prepare_rejects_non_binary(tmp_path):
    src = tmp_path / "notes.md"
    src.write_text("# notes", encoding="utf-8")
    out = tmp_path / "out.md"

    result = runner.invoke(cli_module.app, ["prepare", str(src), "--out", str(out)])

    assert result.exit_code != 0
    assert "supports" in result.output.lower()
