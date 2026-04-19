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
                time.sleep(poll_interval)
        except Exception as exc:
            log.error("worker_loop_error", action="loop_error", error=str(exc), exc_info=True)
            time.sleep(poll_interval)

    log.info("worker_stopped", action="shutdown")
