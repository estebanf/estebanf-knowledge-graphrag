"""Integration tests for Phase 1 ingestion pipeline.

Each test ingests a real file, verifies the result, then cleans up
unconditionally (regardless of pass/fail) so no leftover data remains.
"""

import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import psycopg
import pytest

from rag.config import settings
from rag.ingestion import ingest_file
from rag.profiling import _DEFAULT_PROFILE

TEST_DOCS = Path(__file__).parent.parent / "test_documents"


def cleanup(source_id: str) -> None:
    """Hard-delete source + jobs from DB and remove stored file from disk."""
    url = settings.POSTGRES_URL
    with psycopg.connect(url) as conn:
        conn.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
        conn.execute("DELETE FROM jobs WHERE source_id = %s", (source_id,))
        conn.execute("DELETE FROM sources WHERE id = %s", (source_id,))
        conn.commit()
    stored = settings.STORAGE_BASE_PATH / source_id
    if stored.exists():
        shutil.rmtree(stored)


@pytest.fixture()
def ingested(request):
    """Yield nothing; after each test hard-delete any source_id stored in result."""
    result: dict = {}
    yield result
    if "source_id" in result:
        cleanup(result["source_id"])


@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_ingest_markdown(mock_profile, mock_chunk, mock_validate, mock_embed, ingested):
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True

    file = TEST_DOCS / "Play 2.md"
    result = ingest_file(file, name="test-markdown")
    ingested.update(result)

    assert result["status"] == "completed"
    assert result["source_id"]

    url = settings.POSTGRES_URL
    with psycopg.connect(url) as conn:
        row = conn.execute(
            "SELECT file_type, markdown_content FROM sources WHERE id = %s",
            (result["source_id"],),
        ).fetchone()

    assert row is not None
    assert row[0] == "md"
    assert row[1] and len(row[1]) > 0


@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_ingest_pdf(mock_profile, mock_chunk, mock_validate, mock_embed, ingested):
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True

    file = TEST_DOCS / "Product Leader Insights_ Healthcare Provider Security Buying Behavior.pdf"
    result = ingest_file(file, name="test-pdf")
    ingested.update(result)

    assert result["status"] == "completed"

    url = settings.POSTGRES_URL
    with psycopg.connect(url) as conn:
        row = conn.execute(
            "SELECT file_type, markdown_content FROM sources WHERE id = %s",
            (result["source_id"],),
        ).fetchone()

    assert row is not None
    assert row[0] == "pdf"
    assert row[1] and len(row[1]) > 0


@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_ingest_docx(mock_profile, mock_chunk, mock_validate, mock_embed, ingested):
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True

    file = TEST_DOCS / "Extension GTM Doc.docx"
    result = ingest_file(file, name="test-docx")
    ingested.update(result)

    assert result["status"] == "completed"

    url = settings.POSTGRES_URL
    with psycopg.connect(url) as conn:
        row = conn.execute(
            "SELECT file_type, markdown_content FROM sources WHERE id = %s",
            (result["source_id"],),
        ).fetchone()

    assert row is not None
    assert row[0] == "docx"
    assert row[1] and len(row[1]) > 0


@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_ingest_txt(mock_profile, mock_chunk, mock_validate, mock_embed, ingested, tmp_path):
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True

    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("This is a plain text document.\nSecond line.\n")

    result = ingest_file(txt_file, name="test-txt")
    ingested.update(result)

    assert result["status"] == "completed"

    url = settings.POSTGRES_URL
    with psycopg.connect(url) as conn:
        row = conn.execute(
            "SELECT file_type, markdown_content FROM sources WHERE id = %s",
            (result["source_id"],),
        ).fetchone()

    assert row is not None
    assert row[0] == "txt"
    assert row[1] and len(row[1]) > 0


@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_duplicate_rejected(mock_profile, mock_chunk, mock_validate, mock_embed, ingested):
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True

    file = TEST_DOCS / "Play 2.md"
    result = ingest_file(file, name="test-dedup")
    ingested.update(result)

    with pytest.raises(ValueError, match="Duplicate"):
        ingest_file(file, name="test-dedup-again")


@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_file_stored_on_disk(mock_profile, mock_chunk, mock_validate, mock_embed, ingested):
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True

    file = TEST_DOCS / "Play 2.md"
    result = ingest_file(file, name="test-disk")
    ingested.update(result)

    stored = settings.STORAGE_BASE_PATH / result["source_id"] / file.name
    assert stored.exists()
