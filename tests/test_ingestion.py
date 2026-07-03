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


def cleanup_existing_hash(file_md5: str) -> None:
    """Remove any sources/jobs stored under a given md5 (prepared binary provenance)."""
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


@patch("rag.ingestion.extract_and_store_graph")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_ingest_markdown(mock_profile, mock_chunk, mock_validate, mock_embed, mock_gd, mock_extract, ingested):
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


def _prepare_to_markdown(binary_path: Path, tmp_path: Path) -> tuple[Path, "object"]:
    """Prepare a real binary into markdown (as the CLI would) for an end-to-end
    ingestion test. Image descriptions are stubbed locally so the test needs no
    network. Returns the prepared markdown path and the PreparedDocument."""
    from rag.prepare import finalize_markdown, prepare_binary

    prepared = prepare_binary(binary_path)
    content = finalize_markdown(prepared, lambda data, mime: "[image description]")
    md_path = tmp_path / (binary_path.stem + ".md")
    md_path.write_text(content, encoding="utf-8")
    return md_path, prepared


@patch("rag.ingestion.extract_and_store_graph")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_ingest_prepared_pdf_markdown(mock_profile, mock_chunk, mock_validate, mock_embed, mock_gd, mock_extract, ingested, tmp_path):
    # New contract: the CLI prepares the PDF into markdown; the worker ingests the
    # markdown and the source records original PDF provenance (R9-R11).
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True
    mock_gd.return_value.__enter__ = lambda s: s
    mock_gd.return_value.__exit__ = lambda s, *a: None
    mock_gd.return_value.session.return_value.__enter__ = lambda s: s
    mock_gd.return_value.session.return_value.__exit__ = lambda s, *a: None

    binary = TEST_DOCS / "Product Leader Insights_ Healthcare Provider Security Buying Behavior.pdf"
    md_path, prepared = _prepare_to_markdown(binary, tmp_path)
    cleanup_existing_hash(prepared.original_md5)
    result = ingest_file(
        md_path,
        name="test-pdf",
        original_md5=prepared.original_md5,
        original_file_name=prepared.original_filename,
        original_file_type=prepared.original_extension,
    )
    ingested.update(result)

    assert result["status"] == "completed"

    url = settings.POSTGRES_URL
    with psycopg.connect(url) as conn:
        row = conn.execute(
            "SELECT file_type, markdown_content, md5, metadata FROM sources WHERE id = %s",
            (result["source_id"],),
        ).fetchone()

    assert row is not None
    assert row[0] == "pdf"
    assert row[1] and len(row[1]) > 0
    assert row[2] == prepared.original_md5  # dedup keyed on original binary hash
    assert row[3]["original_filename"] == prepared.original_filename


@patch("rag.ingestion.extract_and_store_graph")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_ingest_prepared_docx_markdown(mock_profile, mock_chunk, mock_validate, mock_embed, mock_gd, mock_extract, ingested, tmp_path):
    mock_profile.return_value = _DEFAULT_PROFILE
    mock_chunk.return_value = []
    mock_validate.return_value = True
    mock_gd.return_value.__enter__ = lambda s: s
    mock_gd.return_value.__exit__ = lambda s, *a: None
    mock_gd.return_value.session.return_value.__enter__ = lambda s: s
    mock_gd.return_value.session.return_value.__exit__ = lambda s, *a: None

    binary = TEST_DOCS / "Extension GTM Doc.docx"
    md_path, prepared = _prepare_to_markdown(binary, tmp_path)
    cleanup_existing_hash(prepared.original_md5)
    result = ingest_file(
        md_path,
        name="test-docx",
        original_md5=prepared.original_md5,
        original_file_name=prepared.original_filename,
        original_file_type=prepared.original_extension,
    )
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


@patch("rag.ingestion.extract_and_store_graph")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_ingest_txt(mock_profile, mock_chunk, mock_validate, mock_embed, mock_gd, mock_extract, ingested, tmp_path):
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


@patch("rag.ingestion.extract_and_store_graph")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_duplicate_rejected(mock_profile, mock_chunk, mock_validate, mock_embed, mock_gd, mock_extract, ingested):
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


@patch("rag.ingestion.extract_and_store_graph")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.embed_and_store_chunks")
@patch("rag.ingestion.validate_chunks")
@patch("rag.ingestion.chunk_document")
@patch("rag.ingestion.profile_document")
def test_file_stored_on_disk(mock_profile, mock_chunk, mock_validate, mock_embed, mock_gd, mock_extract, ingested):
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
