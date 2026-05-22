"""Job management routes: list, status, stats, retry, cancel."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag.db import get_connection
from rag.ingestion import cancel_job, retry_job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobSummary(BaseModel):
    id: str
    source_id: Optional[str]
    status: str
    current_stage: Optional[str]
    stage_log: dict[str, Any]
    created_at: Optional[str]
    updated_at: Optional[str]


class JobListResponse(BaseModel):
    jobs: list[JobSummary]


class JobStatus(JobSummary):
    error_detail: Optional[dict[str, Any]] = None


class JobStatsItem(BaseModel):
    status: str
    count: int


class JobStatsResponse(BaseModel):
    stats: list[JobStatsItem]


class RetryRequest(BaseModel):
    from_stage: Optional[str] = None


def _row_to_summary(row: tuple) -> JobSummary:
    return JobSummary(
        id=str(row[0]),
        source_id=str(row[1]) if row[1] is not None else None,
        status=row[2],
        current_stage=row[3],
        stage_log=row[4] or {},
        created_at=row[5].isoformat() if row[5] else None,
        updated_at=row[6].isoformat() if row[6] else None,
    )


@router.get("/stats", response_model=JobStatsResponse)
def get_job_stats() -> JobStatsResponse:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT
                 CASE
                   WHEN status LIKE 'failed:%' THEN 'failed'
                   WHEN status LIKE 'processing:%' THEN 'processing'
                   ELSE status
                 END AS status_group,
                 COUNT(*) AS cnt
               FROM jobs
               GROUP BY status_group
               ORDER BY status_group"""
        ).fetchall()
    return JobStatsResponse(stats=[JobStatsItem(status=s, count=c) for s, c in rows])


@router.get("", response_model=JobListResponse)
def list_jobs(status: Optional[str] = None) -> JobListResponse:
    with get_connection() as conn:
        if status:
            if status in ("failed", "processing"):
                rows = conn.execute(
                    "SELECT id, source_id, status, current_stage, stage_log, created_at, updated_at "
                    "FROM jobs WHERE status LIKE %s ORDER BY created_at DESC",
                    (f"{status}:%",),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, source_id, status, current_stage, stage_log, created_at, updated_at "
                    "FROM jobs WHERE status = %s ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, source_id, status, current_stage, stage_log, created_at, updated_at "
                "FROM jobs ORDER BY updated_at DESC LIMIT 50"
            ).fetchall()
    return JobListResponse(jobs=[_row_to_summary(r) for r in rows])


@router.get("/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, source_id, status, current_stage, stage_log, created_at, updated_at, error_detail "
            "FROM jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    base = _row_to_summary(row[:7])
    return JobStatus(**base.model_dump(), error_detail=row[7])


@router.post("/{job_id}/retry")
def post_retry(job_id: str, payload: RetryRequest | None = None) -> dict:
    from_stage = payload.from_stage if payload else None
    try:
        return retry_job(job_id, from_stage=from_stage)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{job_id}/cancel")
def post_cancel(job_id: str) -> dict:
    try:
        return cancel_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
