"""Job management routes: list, status, stats, retry, cancel."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag.db import get_connection
from rag.ingestion import cancel_job, retry_job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# Bound on `--days`/`?days=` for the stage-stats/baseline routes (U6/R10):
# an unbounded window risks an expensive jsonb_each aggregation scan over the
# full jobs table (flagged in this plan's security-lens review).
MAX_STAGE_STATS_DAYS = 365

# NOTE: this is a per-stage *duration* aggregation (R8/R10), distinct from
# `GET /stats` above (job *status* counts, consumed by `rag jobs list
# --stats`). Do not merge these two concepts or their schemas.
STAGE_STATS_SQL = """
    SELECT stage,
           count(*) AS job_count,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50_ms,
           percentile_cont(0.9) WITHIN GROUP (ORDER BY duration_ms) AS p90_ms,
           max(duration_ms) AS max_ms
    FROM (
        SELECT kv.key AS stage, (kv.value ->> 'duration_ms')::bigint AS duration_ms
        FROM jobs j, jsonb_each(j.stage_log) kv
        WHERE j.status = 'completed'
          AND j.created_at >= now() - make_interval(days => %s)
          AND kv.value ->> 'duration_ms' IS NOT NULL
    ) sub
    GROUP BY stage
    ORDER BY stage
"""


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


class StageStatItem(BaseModel):
    stage: str
    job_count: int
    p50_ms: Optional[float] = None
    p90_ms: Optional[float] = None
    max_ms: Optional[int] = None


class StageStatsResponse(BaseModel):
    days: int
    stats: list[StageStatItem]


class StageBaselineItem(BaseModel):
    stage: str
    baseline_ms: int
    set_at: Optional[str] = None


class StageBaselineResponse(BaseModel):
    days: int
    baselines: list[StageBaselineItem]


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


def _validate_days(days: int) -> None:
    if days < 1 or days > MAX_STAGE_STATS_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"days must be between 1 and {MAX_STAGE_STATS_DAYS}",
        )


@router.get("/stage-stats", response_model=StageStatsResponse)
def get_job_stage_stats(days: int = 14) -> StageStatsResponse:
    """Per-stage duration_ms percentiles/job count over a time window (R8/R10).

    Computed from completed jobs' `stage_log`; stage entries without a
    `duration_ms` (legacy jobs predating this telemetry) are excluded from the
    aggregation rather than crashing or being coerced to zero.
    """
    _validate_days(days)
    with get_connection() as conn:
        rows = conn.execute(STAGE_STATS_SQL, (days,)).fetchall()
    return StageStatsResponse(
        days=days,
        stats=[
            StageStatItem(stage=r[0], job_count=r[1], p50_ms=r[2], p90_ms=r[3], max_ms=r[4])
            for r in rows
        ],
    )


@router.post("/stage-stats/baseline", response_model=StageBaselineResponse)
def post_stage_stats_baseline(days: int = 14) -> StageBaselineResponse:
    """Snapshot the current window's per-stage median duration as the pinned
    drift-guardrail baseline (R11).

    This OVERWRITES any existing baseline for each stage seen in the window —
    it is an explicit, operator-invoked snapshot (`rag jobs stats
    --set-baseline`), not a value the worker recomputes automatically. Stages
    with no data in the window are left untouched.
    """
    _validate_days(days)
    with get_connection() as conn:
        rows = conn.execute(STAGE_STATS_SQL, (days,)).fetchall()
        stages_written: list[str] = []
        for stage, _job_count, p50_ms, _p90_ms, _max_ms in rows:
            if p50_ms is None:
                continue
            conn.execute(
                """INSERT INTO stage_duration_baseline (stage, baseline_ms, set_at)
                   VALUES (%s, %s, now())
                   ON CONFLICT (stage) DO UPDATE
                     SET baseline_ms = EXCLUDED.baseline_ms, set_at = now()""",
                (stage, int(round(p50_ms))),
            )
            stages_written.append(stage)
        conn.commit()
        if stages_written:
            baseline_rows = conn.execute(
                "SELECT stage, baseline_ms, set_at FROM stage_duration_baseline "
                "WHERE stage = ANY(%s) ORDER BY stage",
                (stages_written,),
            ).fetchall()
        else:
            baseline_rows = []
    return StageBaselineResponse(
        days=days,
        baselines=[
            StageBaselineItem(
                stage=r[0],
                baseline_ms=r[1],
                set_at=r[2].isoformat() if hasattr(r[2], "isoformat") else r[2],
            )
            for r in baseline_rows
        ],
    )


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
