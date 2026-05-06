from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from rag.api.schemas import SourceDetail, SourceInsightsResponse, SourceListResponse
from rag.sources import get_source_detail, list_recent_sources, list_source_insights


router = APIRouter(prefix="/api/sources", tags=["sources"])


@router.get("", response_model=SourceListResponse)
def get_sources(
    limit: int = Query(default=20, gt=0, le=100),
    offset: int = Query(default=0, ge=0),
    metadata: list[str] = Query(default_factory=list),
) -> SourceListResponse:
    metadata_filters = [_parse_metadata_filter(item) for item in metadata]
    result = list_recent_sources(
        limit=limit,
        offset=offset,
        metadata_filters=[item for item in metadata_filters if item is not None],
    )
    return SourceListResponse(
        sources=result["sources"],
        total=result["total"],
        limit=limit,
        offset=offset,
    )


def _parse_metadata_filter(value: str) -> tuple[str, str] | None:
    key, separator, raw = value.partition(":")
    key = key.strip()
    raw = raw.strip()
    if not separator or not key or not raw:
        return None
    return key, raw


@router.get("/{source_id}", response_model=SourceDetail)
def get_source(source_id: str) -> SourceDetail:
    detail = get_source_detail(source_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return SourceDetail(**detail)


@router.get("/{source_id}/insights", response_model=SourceInsightsResponse)
def get_source_insights(source_id: str) -> SourceInsightsResponse:
    return SourceInsightsResponse(insights=list_source_insights(source_id))


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
