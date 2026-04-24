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
from rag.ingestion import compute_md5, ingest_file
from rag.profiling import _DEFAULT_PROFILE

TEST_DOCS = Path(__file__).parent.parent / "test_documents"


def cleanup(source_id: str) -> None:
    """Hard-delete source + jobs from DB and remove stored file from disk."""
    url = settings.POSTGRES_URL
    with psycopg.connect(url) as conn:
        conn.execute("DELETE FROM entities WHERE source_id = %s", (source_id,))
        conn.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
        conn.execute("DELETE FROM jobs WHERE source_id = %s", (source_id,))
        conn.execute("DELETE FROM sources WHERE id = %s", (source_id,))
        conn.commit()
    stored = settings.STORAGE_BASE_PATH / source_id
    if stored.exists():
        shutil.rmtree(stored)


def cleanup_existing_file(file_path: Path) -> None:
    """Remove any active sources/jobs for a file from prior failed test runs."""
    file_md5 = compute_md5(file_path)
    url = settings.POSTGRES_URL
    with psycopg.connect(url) as conn:
        source_rows = conn.execute(
            "SELECT id FROM sources WHERE md5 = %s",
            (file_md5,),
        ).fetchall()
        for row in source_rows:
            source_id = str(row[0])
            conn.execute("DELETE FROM entities WHERE source_id = %s", (source_id,))
            conn.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
            conn.execute("DELETE FROM jobs WHERE source_id = %s", (source_id,))
            conn.execute("DELETE FROM sources WHERE id = %s", (source_id,))
            stored = settings.STORAGE_BASE_PATH / source_id
            if stored.exists():
                shutil.rmtree(stored)
        conn.commit()


@pytest.fixture()
def ingested(request):
    """Yield nothing; after each test hard-delete any source_id stored in result."""
    result: dict = {}
    yield result
    if "source_id" in result:
        cleanup(result["source_id"])


@patch("rag.ingestion.link_graph")
@patch("rag.ingestion.extract_and_store_graph")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_ingest_markdown(mock_profile, mock_chunk, mock_validate, mock_embed, mock_gd, mock_extract, mock_link, ingested):
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True
    mock_gd.return_value.__enter__ = lambda s: s
    mock_gd.return_value.__exit__ = lambda s, *a: None
    mock_gd.return_value.session.return_value.__enter__ = lambda s: s
    mock_gd.return_value.session.return_value.__exit__ = lambda s, *a: None

    file = TEST_DOCS / "Play 2.md"
    cleanup_existing_file(file)
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


@patch("rag.parser.describe_image", return_value="[image description]")
@patch("rag.ingestion.link_graph")
@patch("rag.ingestion.extract_and_store_graph")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_ingest_pdf(mock_profile, mock_chunk, mock_validate, mock_embed, mock_gd, mock_extract, mock_link, mock_describe, ingested):
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True
    mock_gd.return_value.__enter__ = lambda s: s
    mock_gd.return_value.__exit__ = lambda s, *a: None
    mock_gd.return_value.session.return_value.__enter__ = lambda s: s
    mock_gd.return_value.session.return_value.__exit__ = lambda s, *a: None

    file = TEST_DOCS / "Product Leader Insights_ Healthcare Provider Security Buying Behavior.pdf"
    cleanup_existing_file(file)
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


@patch("rag.parser.describe_image", return_value="[image description]")
@patch("rag.ingestion.link_graph")
@patch("rag.ingestion.extract_and_store_graph")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_ingest_docx(mock_profile, mock_chunk, mock_validate, mock_embed, mock_gd, mock_extract, mock_link, mock_describe, ingested):
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True
    mock_gd.return_value.__enter__ = lambda s: s
    mock_gd.return_value.__exit__ = lambda s, *a: None
    mock_gd.return_value.session.return_value.__enter__ = lambda s: s
    mock_gd.return_value.session.return_value.__exit__ = lambda s, *a: None

    file = TEST_DOCS / "Extension GTM Doc.docx"
    cleanup_existing_file(file)
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


@patch("rag.ingestion.link_graph")
@patch("rag.ingestion.extract_and_store_graph")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_ingest_txt(mock_profile, mock_chunk, mock_validate, mock_embed, mock_gd, mock_extract, mock_link, ingested, tmp_path):
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True
    mock_gd.return_value.__enter__ = lambda s: s
    mock_gd.return_value.__exit__ = lambda s, *a: None
    mock_gd.return_value.session.return_value.__enter__ = lambda s: s
    mock_gd.return_value.session.return_value.__exit__ = lambda s, *a: None

    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("This is a plain text document.\nSecond line.\n")
    cleanup_existing_file(txt_file)

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


@patch("rag.ingestion.link_graph")
@patch("rag.ingestion.extract_and_store_graph")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_duplicate_rejected(mock_profile, mock_chunk, mock_validate, mock_embed, mock_gd, mock_extract, mock_link, ingested):
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True
    mock_gd.return_value.__enter__ = lambda s: s
    mock_gd.return_value.__exit__ = lambda s, *a: None
    mock_gd.return_value.session.return_value.__enter__ = lambda s: s
    mock_gd.return_value.session.return_value.__exit__ = lambda s, *a: None

    file = TEST_DOCS / "Play 2.md"
    cleanup_existing_file(file)
    result = ingest_file(file, name="test-dedup")
    ingested.update(result)

    with pytest.raises(ValueError, match="Duplicate"):
        ingest_file(file, name="test-dedup-again")


@patch("rag.ingestion.link_graph")
@patch("rag.ingestion.extract_and_store_graph")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_file_stored_on_disk(mock_profile, mock_chunk, mock_validate, mock_embed, mock_gd, mock_extract, mock_link, ingested):
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True
    mock_gd.return_value.__enter__ = lambda s: s
    mock_gd.return_value.__exit__ = lambda s, *a: None
    mock_gd.return_value.session.return_value.__enter__ = lambda s: s
    mock_gd.return_value.session.return_value.__exit__ = lambda s, *a: None

    file = TEST_DOCS / "Play 2.md"
    cleanup_existing_file(file)
    result = ingest_file(file, name="test-disk")
    ingested.update(result)

    stored = settings.STORAGE_BASE_PATH / result["source_id"] / "1" / f"original_{file.name}"
    assert stored.exists()
