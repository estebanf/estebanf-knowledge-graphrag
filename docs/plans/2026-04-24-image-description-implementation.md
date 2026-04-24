# Image Description Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `<!-- image -->` placeholders (PDF/DOCX/PPTX) and local markdown image references with LLM-generated narrative descriptions during the parsing stage, plus a remediation script for the 126 existing affected sources.

**Architecture:** A new `image_description.py` module handles the OpenRouter multimodal API call. `parser.py` calls it after docling converts the document (for binary formats) or after reading the file text (for markdown). For markdown files, `storage.py` copies referenced image files into versioned storage at submission time so they're available when the async worker parses.

**Tech Stack:** `requests` (already used), `docling` (already used, enable `generate_picture_images=True`), `Pillow` (docling dependency, already available), `re` (stdlib)

**Design doc:** `docs/plans/2026-04-24-image-description-design.md`

---

### Task 1: Add MODEL_IMAGE_DESCRIPTION to config

**Files:**
- Modify: `src/rag/config.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_model_image_description_has_default():
    from rag.config import settings
    assert settings.MODEL_IMAGE_DESCRIPTION == "google/gemini-2.0-flash-lite"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py::test_model_image_description_has_default -v
```

Expected: FAIL with `AttributeError`

**Step 3: Add the config field**

In `src/rag/config.py`, add after `MODEL_ENTITY_EXTRACTION`:

```python
MODEL_IMAGE_DESCRIPTION: str = "google/gemini-2.0-flash-lite"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py::test_model_image_description_has_default -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/rag/config.py tests/test_config.py
git commit -m "feat: add MODEL_IMAGE_DESCRIPTION config key"
```

---

### Task 2: Create image_description module

**Files:**
- Create: `src/rag/image_description.py`
- Create: `tests/test_image_description.py`

**Step 1: Write the failing tests**

Create `tests/test_image_description.py`:

```python
import base64
from unittest.mock import MagicMock, patch

import pytest

from rag.image_description import describe_image


@patch("rag.image_description.requests.post")
def test_describe_image_returns_description(mock_post):
    mock_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "A bar chart showing sales data."}}]
    }
    mock_post.return_value.raise_for_status = MagicMock()

    result = describe_image(b"\x89PNG\r\n", "image/png")

    assert result == "A bar chart showing sales data."


@patch("rag.image_description.requests.post")
def test_describe_image_sends_base64_encoded_image(mock_post):
    mock_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "description"}}]
    }
    mock_post.return_value.raise_for_status = MagicMock()
    image_bytes = b"fake-image-bytes"

    describe_image(image_bytes, "image/jpeg")

    call_json = mock_post.call_args.kwargs["json"]
    content = call_json["messages"][0]["content"]
    image_block = next(b for b in content if b["type"] == "image_url")
    expected_b64 = base64.b64encode(image_bytes).decode()
    assert f"data:image/jpeg;base64,{expected_b64}" in image_block["image_url"]["url"]


@patch("rag.image_description.requests.post")
def test_describe_image_uses_configured_model(mock_post, monkeypatch):
    monkeypatch.setattr("rag.image_description.settings.MODEL_IMAGE_DESCRIPTION", "test/model")
    mock_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    mock_post.return_value.raise_for_status = MagicMock()

    describe_image(b"bytes", "image/png")

    assert mock_post.call_args.kwargs["json"]["model"] == "test/model"


@patch("rag.image_description.requests.post")
def test_describe_image_raises_on_http_error(mock_post):
    import requests as req
    mock_post.return_value.raise_for_status.side_effect = req.HTTPError("500 Server Error")

    with pytest.raises(req.HTTPError):
        describe_image(b"bytes", "image/png")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_image_description.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create the module**

Create `src/rag/image_description.py`:

```python
import base64

import requests

from rag.config import settings

_PROMPT = (
    "Describe this image concisely for use in a knowledge base. "
    "Focus on the key information, data, concepts, or visual content shown. "
    "Be factual and specific. Do not add interpretation beyond what is visible."
)


def describe_image(image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.MODEL_IMAGE_DESCRIPTION,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        },
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
        },
        timeout=(10, 60),
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_image_description.py -v
```

Expected: 4 PASS

**Step 5: Commit**

```bash
git add src/rag/image_description.py tests/test_image_description.py
git commit -m "feat: add image_description module for multimodal LLM calls"
```

---

### Task 3: Describe images in PDF/DOCX/PPTX during parsing

**Files:**
- Modify: `src/rag/parser.py`
- Modify: `tests/test_parser.py`

**Step 1: Write the failing test**

Add to `tests/test_parser.py`:

```python
from unittest.mock import patch


@patch("rag.parser.describe_image", return_value="A diagram showing a workflow.")
def test_parse_document_replaces_image_placeholders_for_pptx(mock_describe):
    file_path = TEST_DOCS / "From AI ambition to a decision-ready roadmap.pptx"

    result = parse_document(file_path)

    assert "<!-- image -->" not in result.markdown
    if mock_describe.called:
        assert "A diagram showing a workflow." in result.markdown
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_parser.py::test_parse_document_replaces_image_placeholders_for_pptx -v
```

Expected: FAIL — `<!-- image -->` still present in markdown

**Step 3: Modify parser.py**

Update `src/rag/parser.py` to enable picture images and replace placeholders:

```python
import io
from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from rag.image_description import describe_image


class ParseError(Exception):
    pass


@dataclass(frozen=True)
class ParseResult:
    markdown: str
    element_tree: str


_TXT_EXTENSIONS = {".txt", ".text"}
_MARKDOWN_EXTENSIONS = {".md", ".markdown"}

_pdf_options = PdfPipelineOptions()
_pdf_options.do_ocr = True
_pdf_options.do_table_structure = True
_pdf_options.generate_page_images = False
_pdf_options.generate_picture_images = True

_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_options),
    }
)


def _plaintext_to_element_tree(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    nonempty = [line for line in lines if line]

    tree_lines = ["0: document with name=_root_"]
    if nonempty and nonempty[0].startswith("#"):
        title = nonempty[0].lstrip("#").strip()
        tree_lines.append(f" 1: title: {title}")
        body = nonempty[1:]
    else:
        body = nonempty

    for index, line in enumerate(body, start=2):
        tree_lines.append(f" {index}: paragraph: {line}")

    return "\n".join(tree_lines)


def _describe_docling_pictures(result, markdown: str) -> str:
    for picture in result.document.pictures:
        img = picture.get_image(result.document)
        if img is None:
            continue
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        description = describe_image(buf.getvalue(), "image/png")
        markdown = markdown.replace("<!-- image -->", description, 1)
    return markdown


def parse_document(file_path: Path) -> ParseResult:
    try:
        suffix = file_path.suffix.lower()
        if suffix in _TXT_EXTENSIONS | _MARKDOWN_EXTENSIONS:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            return ParseResult(markdown=text, element_tree=_plaintext_to_element_tree(text))

        result = _converter.convert(str(file_path))
        markdown = result.document.export_to_markdown()
        markdown = _describe_docling_pictures(result, markdown)
        return ParseResult(
            markdown=markdown,
            element_tree=result.document.export_to_element_tree(),
        )
    except Exception as exc:
        raise ParseError(f"Failed to parse {file_path.name}: {exc}") from exc


def parse_to_markdown(file_path: Path) -> str:
    return parse_document(file_path).markdown
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parser.py -v
```

Expected: all PASS (the new test passes if the PPTX has no images, or verifies replacement if it does)

**Step 5: Commit**

```bash
git add src/rag/parser.py tests/test_parser.py
git commit -m "feat: describe images in PDF/DOCX/PPTX during parsing"
```

---

### Task 4: Store local markdown images at submission time

**Files:**
- Modify: `src/rag/storage.py`
- Modify: `tests/test_storage.py`

**Step 1: Write the failing tests**

Add to `tests/test_storage.py`:

```python
import re

from rag.storage import store_markdown_images


def test_store_markdown_images_copies_referenced_images(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.storage.settings.STORAGE_BASE_PATH", tmp_path)
    img_dir = tmp_path / "docs"
    img_dir.mkdir()
    img_file = img_dir / "chart.png"
    img_file.write_bytes(b"PNG_DATA")
    md_file = img_dir / "report.md"
    md_file.write_text("# Report\n\n![chart](chart.png)\n", encoding="utf-8")

    store_markdown_images("source-abc", md_file, version=1)

    dest = tmp_path / "source-abc" / "1" / "chart.png"
    assert dest.exists()
    assert dest.read_bytes() == b"PNG_DATA"


def test_store_markdown_images_skips_remote_urls(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.storage.settings.STORAGE_BASE_PATH", tmp_path)
    md_file = tmp_path / "report.md"
    md_file.write_text("![x](https://example.com/img.png)\n", encoding="utf-8")

    store_markdown_images("source-abc", md_file, version=1)

    dest = tmp_path / "source-abc" / "1"
    remote_files = list(dest.glob("**/*.png")) if dest.exists() else []
    assert remote_files == []


def test_store_markdown_images_skips_missing_images(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.storage.settings.STORAGE_BASE_PATH", tmp_path)
    md_file = tmp_path / "report.md"
    md_file.write_text("![x](missing.png)\n", encoding="utf-8")

    # Should not raise
    store_markdown_images("source-abc", md_file, version=1)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_storage.py::test_store_markdown_images_copies_referenced_images tests/test_storage.py::test_store_markdown_images_skips_remote_urls tests/test_storage.py::test_store_markdown_images_skips_missing_images -v
```

Expected: FAIL with `ImportError`

**Step 3: Add the function to storage.py**

```python
import re
import shutil
from pathlib import Path

from rag.config import settings

_IMAGE_REF_RE = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
_REMOTE_PREFIXES = ("http://", "https://", "data:")


def store_file(source_id: str, file_path: Path, version: int = 1) -> Path:
    dest_dir = settings.STORAGE_BASE_PATH / source_id / str(version)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"original_{file_path.name}"
    shutil.copy2(file_path, dest)
    return dest


def store_markdown_images(source_id: str, markdown_path: Path, version: int) -> None:
    content = markdown_path.read_text(encoding="utf-8", errors="replace")
    dest_dir = settings.STORAGE_BASE_PATH / source_id / str(version)
    for match in _IMAGE_REF_RE.finditer(content):
        path_str = match.group(1)
        if path_str.startswith(_REMOTE_PREFIXES):
            continue
        img_src = markdown_path.parent / path_str
        if not img_src.exists():
            continue
        img_dest = dest_dir / path_str
        img_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_src, img_dest)


def delete_stored_file(source_id: str) -> None:
    dest_dir = settings.STORAGE_BASE_PATH / source_id
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_storage.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add src/rag/storage.py tests/test_storage.py
git commit -m "feat: copy local markdown image references into versioned storage"
```

---

### Task 5: Call store_markdown_images at submission time

**Files:**
- Modify: `src/rag/ingestion.py`
- Modify: `tests/test_ingestion_submit.py`

**Step 1: Write the failing test**

Add to `tests/test_ingestion_submit.py`:

```python
from unittest.mock import patch, MagicMock


@patch("rag.ingestion.store_markdown_images")
@patch("rag.ingestion.store_file")
@patch("rag.ingestion.get_connection")
def test_submit_ingestion_job_copies_markdown_images(mock_conn, mock_store, mock_store_images, tmp_path):
    md_file = tmp_path / "report.md"
    md_file.write_text("# Report\n\n![chart](chart.png)\n", encoding="utf-8")
    mock_store.return_value = tmp_path / "stored.md"
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    mock_conn.return_value.__enter__.return_value = conn

    from rag.ingestion import submit_ingestion_job
    submit_ingestion_job(md_file)

    mock_store_images.assert_called_once_with("source_id_placeholder", md_file, 1)
```

> Note: the exact source_id is a UUID so you can't assert the exact value. Instead assert it was called once with the correct `file_path` and `version`:

```python
@patch("rag.ingestion.store_markdown_images")
@patch("rag.ingestion.store_file")
@patch("rag.ingestion.get_connection")
def test_submit_ingestion_job_copies_markdown_images(mock_conn, mock_store, mock_store_images, tmp_path):
    md_file = tmp_path / "report.md"
    md_file.write_text("# Report\n\n![chart](chart.png)\n", encoding="utf-8")
    mock_store.return_value = tmp_path / "stored.md"
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    mock_conn.return_value.__enter__.return_value = conn

    from rag.ingestion import submit_ingestion_job
    submit_ingestion_job(md_file)

    assert mock_store_images.call_count == 1
    call_args = mock_store_images.call_args
    assert call_args.args[1] == md_file
    assert call_args.args[2] == 1


@patch("rag.ingestion.store_markdown_images")
@patch("rag.ingestion.store_file")
@patch("rag.ingestion.get_connection")
def test_submit_ingestion_job_does_not_copy_images_for_pdf(mock_conn, mock_store, mock_store_images, tmp_path):
    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF")
    mock_store.return_value = tmp_path / "stored.pdf"
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    mock_conn.return_value.__enter__.return_value = conn

    from rag.ingestion import submit_ingestion_job
    submit_ingestion_job(pdf_file)

    mock_store_images.assert_not_called()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ingestion_submit.py::test_submit_ingestion_job_copies_markdown_images tests/test_ingestion_submit.py::test_submit_ingestion_job_does_not_copy_images_for_pdf -v
```

Expected: FAIL

**Step 3: Add the call in ingestion.py**

In `src/rag/ingestion.py`, add the import at the top:

```python
from rag.storage import store_file, store_markdown_images
```

In `submit_ingestion_job`, after `stored_path = store_file(source_id, file_path, version=1)`:

```python
if file_path.suffix.lower() in {".md", ".markdown"}:
    store_markdown_images(source_id, file_path, version=1)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ingestion_submit.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add src/rag/ingestion.py tests/test_ingestion_submit.py
git commit -m "feat: copy local image files into storage when submitting markdown jobs"
```

---

### Task 6: Describe local images in markdown during parsing

**Files:**
- Modify: `src/rag/parser.py`
- Modify: `tests/test_parser.py`

**Step 1: Write the failing tests**

Add to `tests/test_parser.py`:

```python
from unittest.mock import patch
from PIL import Image as PILImage
import io


@patch("rag.parser.describe_image", return_value="A red square.")
def test_parse_document_replaces_local_image_refs_in_markdown(mock_describe, tmp_path):
    img = PILImage.new("RGB", (10, 10), color="red")
    img_path = tmp_path / "chart.png"
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_path.write_bytes(buf.getvalue())

    md_file = tmp_path / "report.md"
    md_file.write_text("# Report\n\n![chart](chart.png)\n", encoding="utf-8")

    result = parse_document(md_file)

    assert "![chart](chart.png)" not in result.markdown
    assert "A red square." in result.markdown
    mock_describe.assert_called_once()


@patch("rag.parser.describe_image")
def test_parse_document_leaves_remote_image_refs_in_markdown(mock_describe, tmp_path):
    md_file = tmp_path / "report.md"
    md_file.write_text("![x](https://example.com/img.png)\n", encoding="utf-8")

    result = parse_document(md_file)

    mock_describe.assert_not_called()
    assert "https://example.com/img.png" in result.markdown


@patch("rag.parser.describe_image")
def test_parse_document_skips_missing_local_images_in_markdown(mock_describe, tmp_path):
    md_file = tmp_path / "report.md"
    md_file.write_text("![x](missing.png)\n", encoding="utf-8")

    result = parse_document(md_file)

    mock_describe.assert_not_called()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_parser.py::test_parse_document_replaces_local_image_refs_in_markdown tests/test_parser.py::test_parse_document_leaves_remote_image_refs_in_markdown tests/test_parser.py::test_parse_document_skips_missing_local_images_in_markdown -v
```

Expected: FAIL — local refs are not replaced

**Step 3: Add markdown image description to parser.py**

Add to `src/rag/parser.py` after the existing imports:

```python
import re

_IMAGE_REF_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_REMOTE_PREFIXES = ("http://", "https://", "data:")
_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
```

Add a new helper function before `parse_document`:

```python
def _describe_markdown_images(text: str, base_dir: Path) -> str:
    def _replace(m: re.Match) -> str:
        path_str = m.group(2)
        if path_str.startswith(_REMOTE_PREFIXES):
            return m.group(0)
        img_path = base_dir / path_str
        if not img_path.exists():
            return m.group(0)
        mime = _MIME_MAP.get(img_path.suffix.lower(), "image/png")
        description = describe_image(img_path.read_bytes(), mime)
        return description

    return _IMAGE_REF_RE.sub(_replace, text)
```

Update the markdown branch in `parse_document`:

```python
if suffix in _TXT_EXTENSIONS | _MARKDOWN_EXTENSIONS:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if suffix in _MARKDOWN_EXTENSIONS:
        text = _describe_markdown_images(text, file_path.parent)
    return ParseResult(markdown=text, element_tree=_plaintext_to_element_tree(text))
```

**Step 4: Run all parser tests**

```bash
pytest tests/test_parser.py -v
```

Expected: all PASS

**Step 5: Run the full test suite to catch regressions**

```bash
pytest tests/ -v --tb=short
```

Expected: all existing tests still PASS

**Step 6: Commit**

```bash
git add src/rag/parser.py tests/test_parser.py
git commit -m "feat: describe local image references in markdown files during parsing"
```

---

### Task 7: Add image placeholder remediation

**Files:**
- Modify: `src/rag/remediation.py`
- Modify: `tests/test_remediation.py`

**Step 1: Write the failing tests**

Add to `tests/test_remediation.py`:

```python
from rag.remediation import remediate_image_source


@patch("rag.remediation._write_audit_log")
@patch("rag.remediation.cleanup_from_stage")
@patch("rag.remediation.verify_cleanup")
def test_remediate_image_source_requeues_completed_job_from_parsing(
    mock_verify_cleanup,
    mock_cleanup,
    mock_audit,
):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("completed",)
    conn.execute.return_value.fetchall.return_value = []

    remediate_image_source(conn, "driver", "source-1", "job-1", "file.pdf")

    mock_cleanup.assert_called_once_with(conn, "driver", "job-1", "source-1", "parsing")
    mock_verify_cleanup.assert_called_once_with(conn, "driver", "source-1")
    update_call = [c for c in conn.execute.call_args_list if "UPDATE jobs" in str(c)]
    assert len(update_call) == 1
    assert "parsing" in str(update_call[0])
    mock_audit.assert_called_once()
    conn.commit.assert_called_once()


@patch("rag.remediation.cleanup_from_stage")
@patch("rag.remediation.verify_cleanup")
def test_remediate_image_source_rejects_non_completed_jobs(mock_verify_cleanup, mock_cleanup):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("failed:parsing",)

    with pytest.raises(RuntimeError, match="not completed"):
        remediate_image_source(conn, "driver", "source-1", "job-1", "file.pdf")

    mock_cleanup.assert_not_called()
    conn.commit.assert_not_called()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_remediation.py::test_remediate_image_source_requeues_completed_job_from_parsing tests/test_remediation.py::test_remediate_image_source_rejects_non_completed_jobs -v
```

Expected: FAIL with `ImportError`

**Step 3: Add to remediation.py**

Append to `src/rag/remediation.py`:

```python
IDENTIFY_IMAGE_PLACEHOLDER_SOURCES_SQL = """
SELECT
    s.id AS source_id,
    j.id AS job_id,
    s.file_name,
    COUNT(DISTINCT c.id) AS image_chunk_count
FROM chunks c
JOIN sources s ON s.id = c.source_id
JOIN jobs j ON j.source_id = s.id
WHERE j.status = 'completed'
  AND c.deleted_at IS NULL
  AND c.content LIKE '%<!-- image -->%'
GROUP BY s.id, j.id, s.file_name
ORDER BY image_chunk_count DESC, s.file_name ASC
"""


@dataclass(frozen=True)
class AffectedImageSource:
    source_id: str
    job_id: str
    file_name: str
    image_chunk_count: int


def get_image_placeholder_sources(
    conn,
    only_source_id: str | None = None,
    limit: int | None = None,
) -> list[AffectedImageSource]:
    rows = conn.execute(IDENTIFY_IMAGE_PLACEHOLDER_SOURCES_SQL).fetchall()
    affected = [
        AffectedImageSource(
            source_id=str(row[0]),
            job_id=str(row[1]),
            file_name=row[2],
            image_chunk_count=int(row[3]),
        )
        for row in rows
    ]
    if only_source_id:
        affected = [s for s in affected if s.source_id == only_source_id]
    if limit is not None:
        affected = affected[:limit]
    return affected


def remediate_image_source(conn, driver, source_id: str, job_id: str, file_name: str) -> None:
    row = conn.execute(
        "SELECT status FROM jobs WHERE id = %s",
        (job_id,),
    ).fetchone()
    if not row:
        raise RuntimeError(f"Job not found: {job_id}")
    if row[0] != "completed":
        raise RuntimeError(f"Job {job_id} is not completed (status: {row[0]})")

    entity_rows = conn.execute(
        "SELECT id FROM entities WHERE source_id = %s",
        (source_id,),
    ).fetchall()
    entity_ids = [str(row[0]) for row in entity_rows]

    cleanup_from_stage(conn, driver, job_id, source_id, "parsing")
    verify_cleanup(conn, driver, source_id)
    verify_graph_entity_cleanup(driver, entity_ids)
    conn.execute(
        """
        UPDATE jobs
        SET status = 'pending',
            current_stage = NULL,
            retry_from_stage = 'parsing',
            error_detail = NULL,
            updated_at = now()
        WHERE id = %s
        """,
        (job_id,),
    )
    _write_audit_log(
        conn,
        "job_retried",
        "job",
        job_id,
        {"from_stage": "parsing", "reason": "image_placeholder_remediation", "file_name": file_name},
    )
    conn.commit()


def remediate_image_placeholders(
    dry_run: bool = False,
    only_source_id: str | None = None,
    limit: int | None = None,
) -> list[AffectedImageSource]:
    with get_connection() as conn:
        affected = get_image_placeholder_sources(conn, only_source_id=only_source_id, limit=limit)

    if dry_run or not affected:
        return affected

    with get_graph_driver() as driver:
        for source in affected:
            with get_connection() as conn:
                remediate_image_source(conn, driver, source.source_id, source.job_id, source.file_name)
    return affected
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_remediation.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add src/rag/remediation.py tests/test_remediation.py
git commit -m "feat: add image placeholder remediation to remediation module"
```

---

### Task 8: Create remediation script

**Files:**
- Create: `scripts/remediate_image_placeholders.py`

**Step 1: Create the script**

```python
#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.remediation import remediate_image_placeholders


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-queue completed jobs that produced chunks with <!-- image --> placeholders."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print affected sources without changing data.")
    parser.add_argument("--source-id", default=None, help="Limit remediation to a single source UUID.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of sources processed.")
    args = parser.parse_args()

    affected = remediate_image_placeholders(
        dry_run=args.dry_run,
        only_source_id=args.source_id,
        limit=args.limit,
    )

    if not affected:
        print("No affected sources found.")
        return 0

    prefix = "DRY RUN - " if args.dry_run else ""
    print(f"{prefix}Found {len(affected)} affected source(s):")
    for source in affected:
        print(
            f"  {source.file_name} "
            f"source={source.source_id} "
            f"job={source.job_id} "
            f"image_chunks={source.image_chunk_count}"
        )
    if not args.dry_run:
        print("Remediation complete. Start the worker to process the re-queued jobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 2: Make it executable**

```bash
chmod +x scripts/remediate_image_placeholders.py
```

**Step 3: Smoke-test with dry-run**

```bash
python scripts/remediate_image_placeholders.py --dry-run --limit 5
```

Expected: prints up to 5 affected sources without modifying any data

**Step 4: Commit**

```bash
git add scripts/remediate_image_placeholders.py
git commit -m "feat: add remediate_image_placeholders script"
```

---

### Task 9: Run full test suite and verify

**Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS (85 existing + new tests)

**Step 2: Run the remediation dry-run against real data**

```bash
python scripts/remediate_image_placeholders.py --dry-run
```

Expected: shows ~126 sources

**Step 3: Final commit if any fixes needed**

```bash
git add -p
git commit -m "fix: address test suite issues"
```
