from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Optional

from rag.community import detect_communities
from rag.config import settings
from rag.db import get_connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_stage(run_id: str, entry: dict) -> None:
    with get_connection() as conn:
        conn.execute(
            """UPDATE community_runs
               SET stage_log = stage_log || %s::jsonb,
                   updated_at = now()
               WHERE id = %s""",
            (json.dumps([entry]), run_id),
        )


def _reap_stale_runs() -> None:
    with get_connection() as conn:
        conn.execute(
            """UPDATE community_runs
               SET status = 'failed',
                   error = 'stale',
                   updated_at = now()
               WHERE status = 'running'
                 AND updated_at < now() - make_interval(secs => %s)""",
            (settings.COMMUNITY_RUN_STALE_SECONDS,),
        )


def create_run(params: dict, source_ids: list[str]) -> str:
    _reap_stale_runs()

    with get_connection() as conn:
        active = conn.execute(
            """SELECT count(*) FROM community_runs
               WHERE status NOT IN ('completed', 'failed')""",
        ).fetchone()[0]
    if active >= settings.COMMUNITY_MAX_CONCURRENT_RUNS:
        raise RuntimeError(
            f"Too many concurrent community runs ({active} active, max {settings.COMMUNITY_MAX_CONCURRENT_RUNS})"
        )

    with get_connection() as conn:
        row = conn.execute(
            """INSERT INTO community_runs (params, source_ids)
               VALUES (%s::jsonb, %s::jsonb)
               RETURNING id""",
            (json.dumps(params), json.dumps(source_ids)),
        ).fetchone()
    return str(row[0])


def execute_run(run_id: str) -> None:
    try:
        _append_stage(run_id, {"stage": "start", "at": _now()})

        with get_connection() as conn:
            row = conn.execute(
                "SELECT params, source_ids FROM community_runs WHERE id = %s",
                (run_id,),
            ).fetchone()
        if row is None:
            return
        params = row[0] if isinstance(row[0], dict) else {}
        source_ids = row[1] if isinstance(row[1], list) else []

        _append_stage(
            run_id,
            {"stage": "resolve_scope", "at": _now(), "counts": {"sources": len(source_ids)}},
        )

        result = detect_communities(
            scope_mode=params.get("scope_mode", "ids"),
            source_ids=source_ids,
            criteria=params.get("criteria", []),
            filters=params.get("filters", {}),
            search_options=params.get("search_options", {}),
            retrieve_options=params.get("retrieve_options", {}),
            semantic_threshold=params.get("semantic_threshold"),
            source_cooc_weight=params.get("source_cooc_weight"),
            cutoff=params.get("cutoff"),
            min_community_size=params.get("min_community_size"),
            top_k_chunks=params.get("top_k_chunks"),
            summarize_model=params.get("summarize_model"),
            cross_source_top_k=params.get("cross_source_top_k"),
            max_cross_source_queries=params.get("max_cross_source_queries"),
            resolution=params.get("resolution"),
        )

        metadata = result.get("metadata", {})
        communities = result.get("communities", [])

        _append_stage(
            run_id,
            {
                "stage": "leiden",
                "at": _now(),
                "counts": {"communities": len(communities)},
            },
        )

        summarized = sum(1 for c in communities if c.get("summary"))
        if summarized:
            _append_stage(
                run_id,
                {
                    "stage": "summaries",
                    "at": _now(),
                    "counts": {"done": summarized, "total": len(communities)},
                },
            )

        with get_connection() as conn:
            conn.execute(
                """UPDATE community_runs
                   SET status = 'completed',
                       result = %s::jsonb,
                       source_ids = %s::jsonb,
                       updated_at = now()
                   WHERE id = %s""",
                (json.dumps(result), json.dumps(source_ids), run_id),
            )
        _append_stage(run_id, {"stage": "completed", "at": _now()})

    except Exception as e:
        error_msg = str(e)
        _append_stage(run_id, {"stage": "failed", "at": _now(), "error": error_msg})
        try:
            with get_connection() as conn:
                conn.execute(
                    """UPDATE community_runs
                       SET status = 'failed',
                           error = %s,
                           updated_at = now()
                       WHERE id = %s""",
                    (error_msg, run_id),
                )
        except Exception:
            pass


def get_run(run_id: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            """SELECT id, status, params, source_ids, stage_log,
                      result, error, created_at, updated_at
               FROM community_runs WHERE id = %s""",
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def list_runs(limit: int = 20, offset: int = 0) -> dict:
    with get_connection() as conn:
        total = conn.execute(
            "SELECT count(*) FROM community_runs"
        ).fetchone()[0]
        rows = conn.execute(
            """SELECT id, status, params, source_ids, stage_log,
                      result, error, created_at, updated_at
               FROM community_runs
               ORDER BY created_at DESC
               LIMIT %s OFFSET %s""",
            (limit, offset),
        ).fetchall()
    return {
        "runs": [_row_to_dict(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def _row_to_dict(row) -> dict:
    return {
        "id": str(row[0]),
        "status": row[1] or "running",
        "params": row[2] if isinstance(row[2], dict) else (json.loads(row[2]) if isinstance(row[2], str) else {}),
        "source_ids": row[3] if isinstance(row[3], list) else [],
        "stage_log": row[4] if isinstance(row[4], list) else [],
        "result": row[5] if isinstance(row[5], dict) else None,
        "error": row[6],
        "created_at": row[7].isoformat() if row[7] else None,
        "updated_at": row[8].isoformat() if row[8] else None,
    }


def stream_run_events(run_id: str, seen_count: int = 0):
    import time
    while True:
        run = get_run(run_id)
        if run is None:
            yield {"event": "error", "data": json.dumps({"message": "run not found"})}
            return

        entries = run.get("stage_log", [])
        if isinstance(entries, list):
            for i in range(seen_count, len(entries)):
                entry = entries[i]
                yield {"event": "stage", "data": json.dumps(entry)}

        seen_count = len(entries) if isinstance(entries, list) else seen_count

        if run["status"] in ("completed", "failed"):
            yield {"event": "result", "data": json.dumps({
                "status": run["status"],
                "result": run.get("result"),
                "error": run.get("error"),
            })}
            return

        time.sleep(0.5)
