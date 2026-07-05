from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from rag.api.schemas import SourceDetail, SourceInsightsResponse, SourceListResponse
from rag.db import get_connection
from rag.graph_db import get_graph_driver
from rag.ingestion import _write_audit_log, delete_source_artifacts
from rag.sources import get_source_detail, list_recent_sources, list_source_insights
from rag.storage import delete_stored_file


router = APIRouter(prefix="/api/sources", tags=["sources"])


@router.get("/facets")
def get_facets() -> dict:
    facet_keys = ["kind", "author", "source", "domain"]
    result: dict[str, list[dict]] = {}
    with get_connection() as conn:
        for key in facet_keys:
            rows = conn.execute(
                """SELECT COALESCE(metadata->>%s, '(none)') AS value, count(*) AS cnt
                   FROM sources WHERE deleted_at IS NULL
                   GROUP BY 1 ORDER BY cnt DESC""",
                (key,),
            ).fetchall()
            result[key] = [{"value": r[0], "count": r[1]} for r in rows]
    return {"facets": result}


@router.get("", response_model=SourceListResponse)
def get_sources(
    limit: int = Query(default=20, gt=0, le=100),
    offset: int = Query(default=0, ge=0),
    metadata: list[str] = Query(default_factory=list),
    q: str | None = Query(default=None),
) -> SourceListResponse:
    metadata_filters = [_parse_metadata_filter(item) for item in metadata]
    result = list_recent_sources(
        limit=limit,
        offset=offset,
        metadata_filters=[item for item in metadata_filters if item is not None],
        q=q,
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


@router.delete("/{source_id}")
def delete_source(source_id: str, hard: bool = Query(default=False)) -> dict:
    """Soft- or hard-delete a source."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT storage_path FROM sources WHERE id = %s AND deleted_at IS NULL",
            (source_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"source not found: {source_id}")

        if hard:
            with get_graph_driver() as driver:
                delete_source_artifacts(conn, driver, source_id)
        else:
            conn.execute(
                "UPDATE sources SET deleted_at = now() WHERE id = %s",
                (source_id,),
            )
        _write_audit_log(
            conn,
            "source_hard_deleted" if hard else "source_soft_deleted",
            "source",
            source_id,
            {"hard": hard},
        )
        conn.commit()

    if hard:
        delete_stored_file(source_id)
    return {"source_id": source_id, "hard": hard}
