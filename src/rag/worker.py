import signal
import time

import psycopg
import psycopg.types.json
import structlog

from rag.config import settings
from rag.db import get_connection
from rag.ingestion import execute_ingestion_pipeline
from rag.logging_config import configure_logging

log = structlog.get_logger()

_running = True


def _set_stopped(signum, frame):
    global _running
    _running = False


def recover_stuck_jobs(conn: psycopg.Connection, stuck_minutes: int) -> int:
    rows = conn.execute(
        """
        SELECT id, current_stage FROM jobs
        WHERE status LIKE 'processing:%%'
          AND updated_at < now() - make_interval(mins => %s)
        FOR UPDATE SKIP LOCKED
        """,
        (stuck_minutes,),
    ).fetchall()
    for row in rows:
        job_id = str(row[0])
        stage = row[1] or "unknown"
        conn.execute(
            """UPDATE jobs SET status = %s, updated_at = now(), error_detail = %s WHERE id = %s""",
            (
                f"failed:{stage}",
                psycopg.types.json.Jsonb({"stage": stage, "message": "Worker crashed or job timed out", "traceback": None}),
                job_id,
            ),
        )
    conn.commit()
    return len(rows)


def check_stage_drift(conn: psycopg.Connection, job_id: str) -> None:
    """Compare a just-completed job's per-stage `duration_ms` against the
    *pinned* baseline in `stage_duration_baseline` (R11).

    Deliberately reads the frozen baseline row, not a recomputed recent-window
    median: a moving baseline drifts upward together with a gradual
    corpus-coupled regression and would never fire for the failure class this
    guardrail exists to catch (the April->June intake slowdown in this plan's
    Problem Frame). A stage with no baseline set yet is skipped — missing
    baseline is not treated as drift.
    """
    row = conn.execute("SELECT stage_log FROM jobs WHERE id = %s", (job_id,)).fetchone()
    if not row or not row[0]:
        return
    stage_log = row[0]

    baseline_rows = conn.execute("SELECT stage, baseline_ms FROM stage_duration_baseline").fetchall()
    baselines = {r[0]: r[1] for r in baseline_rows}
    if not baselines:
        return

    factor = settings.STAGE_DRIFT_WARN_FACTOR
    for stage, entry in stage_log.items():
        if not isinstance(entry, dict):
            continue
        duration_ms = entry.get("duration_ms")
        baseline_ms = baselines.get(stage)
        if duration_ms is None or baseline_ms is None:
            continue
        threshold_ms = baseline_ms * factor
        if duration_ms > threshold_ms:
            log.warning(
                "stage_duration_drift",
                action="drift_check",
                job_id=job_id,
                stage=stage,
                duration_ms=duration_ms,
                baseline_ms=baseline_ms,
                factor=factor,
                threshold_ms=threshold_ms,
            )


def claim_next_job(conn: psycopg.Connection) -> tuple[str, str, str] | None:
    row = conn.execute(
        """
        SELECT id, source_id, retry_from_stage FROM jobs
        WHERE status = 'pending'
        ORDER BY created_at
        LIMIT 1
        FOR UPDATE SKIP LOCKED
        """,
    ).fetchone()
    if not row:
        conn.rollback()
        return None
    job_id = str(row[0])
    source_id = str(row[1])
    start_stage = row[2] or "parsing"
    conn.execute(
        "UPDATE jobs SET status = %s, current_stage = %s, updated_at = now() WHERE id = %s",
        (f"processing:{start_stage}", start_stage, job_id),
    )
    conn.commit()
    return job_id, source_id, start_stage


def run_worker(poll_interval: int | None = None, stuck_minutes: int | None = None) -> None:
    global _running
    _running = True
    poll_interval = poll_interval or settings.WORKER_POLL_INTERVAL
    stuck_minutes = stuck_minutes or settings.WORKER_STUCK_JOB_MINUTES

    configure_logging(settings.LOG_LEVEL)
    signal.signal(signal.SIGINT, _set_stopped)
    signal.signal(signal.SIGTERM, _set_stopped)

    log.info("worker_started", action="startup", poll_interval=poll_interval, stuck_minutes=stuck_minutes)

    with get_connection() as conn:
        recovered = recover_stuck_jobs(conn, stuck_minutes)
    if recovered:
        log.warning("stuck_jobs_recovered", action="recovery", count=recovered)

    while _running:
        try:
            with get_connection() as conn:
                claimed = claim_next_job(conn)
            if claimed:
                job_id, source_id, start_stage = claimed
                log.info("job_claimed", action="claim", job_id=job_id, source_id=source_id, start_stage=start_stage)
                try:
                    execute_ingestion_pipeline(job_id, source_id, start_stage=start_stage)
                except Exception as exc:
                    log.error("pipeline_error", action="pipeline_error", job_id=job_id, error=str(exc), exc_info=True)
                else:
                    try:
                        with get_connection() as drift_conn:
                            check_stage_drift(drift_conn, job_id)
                    except Exception as exc:
                        log.error("drift_check_error", action="drift_check_error", job_id=job_id, error=str(exc), exc_info=True)
            else:
                time.sleep(poll_interval)
        except Exception as exc:
            log.error("worker_loop_error", action="loop_error", error=str(exc), exc_info=True)
            time.sleep(poll_interval)

    log.info("worker_stopped", action="shutdown")
