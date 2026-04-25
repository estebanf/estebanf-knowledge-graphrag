from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from rag.api.schemas import SourceDetail
from rag.sources import get_source_detail


router = APIRouter(prefix="/api/sources", tags=["sources"])


@router.get("/{source_id}", response_model=SourceDetail)
def get_source(source_id: str) -> SourceDetail:
    detail = get_source_detail(source_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return SourceDetail(**detail)


@router.get("/{source_id}/download")
def download_source(source_id: str) -> FileResponse:
    detail = get_source_detail(source_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Source not found")

    stored_path = Path(detail["storage_path"])
    # Relative paths stored by local ingestion resolve against CWD. Inside a
    # container CWD is /app but the volume is mounted at /; try the absolute
    # form as a fallback so both environments work.
    if not stored_path.is_absolute() and not stored_path.exists():
        stored_path = Path("/") / stored_path
    if not stored_path.exists():
        raise HTTPException(status_code=404, detail="Stored file not found")

    return FileResponse(
        path=stored_path,
        media_type=detail.get("file_type") or "application/octet-stream",
        filename=detail.get("file_name") or stored_path.name,
    )
