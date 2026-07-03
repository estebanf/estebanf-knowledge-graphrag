"""Tests for CLI-side binary preparation (rag.prepare)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rag.prepare import (
    ExtractedImage,
    PrepareError,
    PreparedDocument,
    finalize_markdown,
    prepare_binary,
)

TEST_DOCS = Path(__file__).parent.parent / "test_documents"


def test_importing_prepare_does_not_eagerly_import_docling():
    cmd = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "import rag.prepare; "
            "print(json.dumps({'docling': any(n.startswith('docling') for n in sys.modules)}))"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == '{"docling": false}'


def test_prepare_binary_returns_markdown_and_metadata_for_pptx():
    pytest.importorskip("docling")
    file_path = TEST_DOCS / "From AI ambition to a decision-ready roadmap.pptx"

    prepared = prepare_binary(file_path)

    assert isinstance(prepared, PreparedDocument)
    assert prepared.markdown
    assert "decision-ready roadmap" in prepared.markdown.lower()
    assert prepared.original_filename == file_path.name
    assert prepared.original_extension == "pptx"
    assert len(prepared.original_md5) == 32
    assert prepared.image_count == len(prepared.images)


def test_finalize_markdown_replaces_placeholders_in_order():
    # AE2: descriptions are consumed in order, no unresolved placeholders remain.
    prepared = PreparedDocument(
        markdown="# Doc\n\n<!-- image -->\n\ntext\n\n<!-- image -->\n\n<!-- image -->\n",
        images=[
            ExtractedImage(data=b"a", mime_type="image/png"),
            ExtractedImage(data=b"b", mime_type="image/png"),
            ExtractedImage(data=b"c", mime_type="image/png"),
        ],
        original_filename="doc.pdf",
        original_extension="pdf",
        original_md5="0" * 32,
        image_count=3,
    )

    calls: list[bytes] = []

    def describe(data: bytes, mime: str) -> str:
        calls.append(data)
        return f"DESC-{len(calls)}"

    result = finalize_markdown(prepared, describe)

    assert "<!-- image -->" not in result
    assert calls == [b"a", b"b", b"c"]
    assert result.index("DESC-1") < result.index("DESC-2") < result.index("DESC-3")


def test_finalize_markdown_hard_fails_when_describe_raises():
    prepared = PreparedDocument(
        markdown="<!-- image -->",
        images=[ExtractedImage(data=b"a", mime_type="image/png")],
        original_filename="doc.pdf",
        original_extension="pdf",
        original_md5="0" * 32,
        image_count=1,
    )

    def describe(data: bytes, mime: str) -> str:
        raise RuntimeError("backend down")

    with pytest.raises(PrepareError, match="doc.pdf"):
        finalize_markdown(prepared, describe)


def test_finalize_markdown_hard_fails_on_unresolved_placeholder():
    # More placeholders than extracted images -> a dangling marker must hard-fail.
    prepared = PreparedDocument(
        markdown="<!-- image -->\n<!-- image -->",
        images=[ExtractedImage(data=b"a", mime_type="image/png")],
        original_filename="doc.pdf",
        original_extension="pdf",
        original_md5="0" * 32,
        image_count=1,
    )

    with pytest.raises(PrepareError, match="Unresolved image placeholders"):
        finalize_markdown(prepared, lambda data, mime: "DESC")


def test_finalize_markdown_no_images_returns_markdown_unchanged():
    prepared = PreparedDocument(
        markdown="# Just text\n\nNo images here.\n",
        images=[],
        original_filename="doc.pdf",
        original_extension="pdf",
        original_md5="0" * 32,
        image_count=0,
    )

    called = False

    def describe(data: bytes, mime: str) -> str:
        nonlocal called
        called = True
        return "DESC"

    result = finalize_markdown(prepared, describe)

    assert result == prepared.markdown
    assert called is False


@patch("rag.prepare._get_docling_converter", side_effect=PrepareError("prepare extra missing"))
def test_prepare_binary_raises_prepare_error_when_docling_missing(mock_converter, tmp_path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    with pytest.raises(PrepareError, match="prepare extra missing"):
        prepare_binary(file_path)
