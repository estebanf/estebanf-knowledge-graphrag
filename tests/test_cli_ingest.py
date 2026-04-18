from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from rag.cli import app

runner = CliRunner()


@patch("rag.cli.submit_ingestion_job")
def test_ingest_single_file_shows_pending(mock_submit, tmp_path):
    f = tmp_path / "test.pdf"
    f.write_bytes(b"content")
    mock_submit.return_value = {"source_id": "s1", "job_id": "j1", "status": "pending"}

    result = runner.invoke(app, ["ingest", str(f)])

    assert result.exit_code == 0
    assert "pending" in result.output
    assert "j1" in result.output


@patch("rag.cli.submit_ingestion_job")
def test_ingest_multiple_files_shows_table(mock_submit, tmp_path):
    f1 = tmp_path / "file1.pdf"
    f2 = tmp_path / "file2.md"
    f1.write_bytes(b"content1")
    f2.write_bytes(b"content2")

    mock_submit.side_effect = [
        {"source_id": "s1", "job_id": "job-aaa", "status": "pending"},
        {"source_id": "s2", "job_id": "job-bbb", "status": "pending"},
    ]

    result = runner.invoke(app, ["ingest", str(f1), str(f2)])

    assert result.exit_code == 0
    assert "job-aaa" in result.output
    assert "job-bbb" in result.output


@patch("rag.cli.submit_ingestion_job")
def test_ingest_folder_finds_supported_files(mock_submit, tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"pdf content")
    (tmp_path / "notes.txt").write_bytes(b"txt content")
    (tmp_path / "slides.pptx").write_bytes(b"pptx content")

    mock_submit.return_value = {"source_id": "sx", "job_id": "jx", "status": "pending"}

    result = runner.invoke(app, ["ingest", str(tmp_path)])

    assert result.exit_code == 0
    assert mock_submit.call_count == 3


@patch("rag.cli.submit_ingestion_job")
def test_ingest_folder_skips_unsupported(mock_submit, tmp_path):
    (tmp_path / "doc.xyz").write_bytes(b"unsupported content")

    result = runner.invoke(app, ["ingest", str(tmp_path)])

    assert result.exit_code == 0
    assert "No supported files found" in result.output
    mock_submit.assert_not_called()


@patch("rag.cli.submit_ingestion_job")
def test_ingest_folder_empty_dir(mock_submit, tmp_path):
    result = runner.invoke(app, ["ingest", str(tmp_path)])

    assert result.exit_code == 0
    assert "No supported files" in result.output
    mock_submit.assert_not_called()
