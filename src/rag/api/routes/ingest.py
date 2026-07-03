"""Ingest routes: multipart file upload and JSON text submission."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from rag.ingestion import submit_ingestion_job

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

# The backend worker parses markdown/text only; binary documents are prepared on
# the CLI into self-contained markdown before submission (see rag.prepare). Direct
# multipart upload therefore accepts text formats only and rejects binaries with a
# clear message (R12, AE4).
TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
BINARY_EXTENSIONS = {".pdf", ".docx", ".pptx"}


class IngestResponse(BaseModel):
    source_id: str
    job_id: str
    status: str


class IngestTextRequest(BaseModel):
    content: str
    name: Optional[str] = None
    metadata: Optional[dict] = None
    original_md5: Optional[str] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None


def _parse_metadata(raw: Optional[str]) -> Optional[dict]:
    if raw is None or raw == "":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid metadata json: {exc}")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    return parsed


@router.post("", response_model=IngestResponse)
async def ingest_multipart(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
) -> IngestResponse:
    """Accept a multipart upload and submit an ingestion job."""
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix in BINARY_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"binary documents ({suffix}) cannot be uploaded directly: prepare "
                "them on the CLI with `rag ingest` or `rag prepare`, which converts "
                "them to markdown before submission."
            ),
        )
    if suffix not in TEXT_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type: {suffix or 'unknown'}",
        )

    meta = _parse_metadata(metadata)

    # Stream the upload into a temp file the ingestion pipeline can hash + copy.
    tmp = tempfile.NamedTemporaryFile(prefix="ingest-", suffix=Path(filename).name, delete=False)
    tmp_path = Path(tmp.name)
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.close()
        # Use the original filename so storage paths reflect what the user uploaded.
        named_path = tmp_path.with_name(filename)
        tmp_path.rename(named_path)
        tmp_path = named_path
        try:
            result = submit_ingestion_job(tmp_path, name=name, metadata=meta)
        except ValueError as exc:
            message = str(exc)
            status = 409 if message.lower().startswith("duplicate") else 400
            raise HTTPException(status_code=status, detail=message)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return IngestResponse(**result)
    finally:
        # ingestion.store_file copies the file under STORAGE_BASE_PATH, so the
        # temp can go away on success or failure.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


@router.post("/text", response_model=IngestResponse)
def ingest_text(payload: IngestTextRequest) -> IngestResponse:
    """Accept inline text and submit an ingestion job as a markdown document."""
    tmp = tempfile.NamedTemporaryFile(prefix="ingest-", suffix=".md", delete=False, mode="w", encoding="utf-8")
    tmp_path = Path(tmp.name)
    try:
        tmp.write(payload.content)
        tmp.close()
        try:
            result = submit_ingestion_job(
                tmp_path,
                name=payload.name,
                metadata=payload.metadata,
                original_md5=payload.original_md5,
                original_file_name=payload.file_name,
                original_file_type=payload.file_type,
            )
        except ValueError as exc:
            message = str(exc)
            status = 409 if message.lower().startswith("duplicate") else 400
            raise HTTPException(status_code=status, detail=message)
        return IngestResponse(**result)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
