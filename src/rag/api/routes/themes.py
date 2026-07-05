from fastapi import APIRouter, HTTPException

from rag.themes import (
    get_report,
    list_reports,
    start_regenerate_report,
    start_theme_report,
)

router = APIRouter(prefix="/api/themes", tags=["themes"])


@router.get("")
def list_theme_reports_api(
    limit: int = 20,
    offset: int = 0,
) -> dict:
    return list_reports(limit=limit, offset=offset)


@router.post("")
def generate_theme(payload: dict) -> dict:
    run_id = payload.get("run_id")
    if not run_id:
        raise HTTPException(status_code=422, detail="run_id is required")
    model = payload.get("model")
    try:
        report_id = start_theme_report(run_id, model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"id": report_id, "status": "generating"}


@router.get("/{report_id}")
def get_theme_report(report_id: str) -> dict:
    report = get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="theme report not found")
    return report


@router.post("/{report_id}/regenerate")
def regenerate_theme(report_id: str, payload: dict | None = None) -> dict:
    model = (payload or {}).get("model")
    try:
        report_id_new = start_regenerate_report(report_id, model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"id": report_id_new, "status": "generating"}
