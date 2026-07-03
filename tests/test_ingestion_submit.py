from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.ingestion import submit_ingestion_job


@patch("rag.ingestion.get_connection")
@patch("rag.ingestion.store_file")
@patch("rag.ingestion.check_duplicate")
@patch("rag.ingestion.compute_md5")
def test_submit_returns_pending(mock_md5, mock_dup, mock_store, mock_conn, tmp_path):
    f = tmp_path / "test.pdf"
    f.write_bytes(b"content")

    mock_md5.return_value = "abc123"
    mock_dup.return_value = None
    mock_store.return_value = Path("/data/documents/123/test.pdf")

    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn

    result = submit_ingestion_job(f)

    assert result["status"] == "pending"
    assert "job_id" in result
    assert "source_id" in result


@patch("rag.ingestion.get_connection")
@patch("rag.ingestion.compute_md5")
def test_submit_duplicate_raises_value_error(mock_md5, mock_conn, tmp_path):
    f = tmp_path / "test.pdf"
    f.write_bytes(b"content")

    mock_md5.return_value = "abc123"

    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("existing-id",)

    with pytest.raises(ValueError, match="Duplicate"):
        submit_ingestion_job(f)


def test_submit_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        submit_ingestion_job(Path("/nonexistent/file.pdf"))


@patch("rag.ingestion.get_connection")
@patch("rag.ingestion.store_file")
@patch("rag.ingestion.check_duplicate")
@patch("rag.ingestion.compute_md5")
def test_submit_inserts_pending_job(mock_md5, mock_dup, mock_store, mock_conn, tmp_path):
    f = tmp_path / "test.pdf"
    f.write_bytes(b"content")

    mock_md5.return_value = "deadbeef"
    mock_dup.return_value = None
    mock_store.return_value = Path("/data/documents/abc/test.pdf")

    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn

    submit_ingestion_job(f)

    # Collect all SQL strings passed to execute
    all_sql = [str(c[0][0]) for c in conn.execute.call_args_list]
    # The jobs INSERT should contain 'pending'
    jobs_inserts = [s for s in all_sql if "INSERT INTO jobs" in s]
    assert jobs_inserts, "Expected an INSERT INTO jobs call"
    assert any("pending" in s for s in jobs_inserts)


@patch("rag.ingestion.store_markdown_images")
@patch("rag.ingestion.get_connection")
@patch("rag.ingestion.store_file")
@patch("rag.ingestion.check_duplicate")
@patch("rag.ingestion.compute_md5")
def test_submit_ingestion_job_copies_markdown_images(mock_md5, mock_dup, mock_store, mock_conn, mock_store_images, tmp_path):
    md_file = tmp_path / "report.md"
    md_file.write_text("# Report\n\n![chart](chart.png)\n", encoding="utf-8")
    mock_md5.return_value = "abc123"
    mock_dup.return_value = None
    mock_store.return_value = tmp_path / "stored.md"
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn

    submit_ingestion_job(md_file)

    assert mock_store_images.call_count == 1
    call_args = mock_store_images.call_args
    assert call_args.args[1] == md_file
    assert call_args.kwargs.get("version", call_args.args[2] if len(call_args.args) > 2 else None) == 1


@patch("rag.ingestion.get_connection")
@patch("rag.ingestion.store_file")
@patch("rag.ingestion.check_duplicate")
@patch("rag.ingestion.compute_md5")
def test_submit_uses_original_md5_for_dedup_and_columns(mock_md5, mock_dup, mock_store, mock_conn, tmp_path):
    # Prepared binary: submitted file is markdown, but dedup + provenance use the
    # original binary hash and filename (R10, R11).
    prepared = tmp_path / "prepared.md"
    prepared.write_text("# prepared\n\ncontent", encoding="utf-8")
    mock_md5.return_value = "markdown-hash"  # hash of the prepared markdown
    mock_dup.return_value = None
    mock_store.return_value = tmp_path / "stored.md"
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn

    submit_ingestion_job(
        prepared,
        metadata={"prepared_image_count": 2},
        original_md5="pdf-hash",
        original_file_name="report.pdf",
        original_file_type="pdf",
    )

    # Dedup checks the original binary hash, not the markdown hash.
    assert mock_dup.call_args.args[1] == "pdf-hash"
    source_insert = next(
        c for c in conn.execute.call_args_list if "INSERT INTO sources" in str(c[0][0])
    )
    params = source_insert.args[1]
    # (id, name, file_name, file_type, storage_path, md5, metadata)
    assert params[1] == "report"  # name derived from original filename stem
    assert params[2] == "report.pdf"  # file_name column = original
    assert params[3] == "pdf"  # file_type column = original
    assert params[5] == "pdf-hash"  # stored md5 = original binary hash
    stored_metadata = params[6].obj
    assert stored_metadata["original_md5"] == "pdf-hash"
    assert stored_metadata["original_filename"] == "report.pdf"
    assert stored_metadata["original_extension"] == "pdf"
    assert stored_metadata["prepared_image_count"] == 2


@patch("rag.ingestion.store_markdown_images")
@patch("rag.ingestion.get_connection")
@patch("rag.ingestion.store_file")
@patch("rag.ingestion.check_duplicate")
@patch("rag.ingestion.compute_md5")
def test_submit_ingestion_job_does_not_copy_images_for_pdf(mock_md5, mock_dup, mock_store, mock_conn, mock_store_images, tmp_path):
    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF")
    mock_md5.return_value = "abc123"
    mock_dup.return_value = None
    mock_store.return_value = tmp_path / "stored.pdf"
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn

    submit_ingestion_job(pdf_file)

    mock_store_images.assert_not_called()
