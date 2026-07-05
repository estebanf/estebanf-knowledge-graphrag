import json
import threading

from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from rag.api.schemas import CommunityRequest
from rag.community import detect_communities
from rag.community_runs import (
    complete_run,
    create_run,
    execute_run,
    get_run,
    list_runs,
    stream_run_events,
)
from rag.db import get_connection

router = APIRouter(prefix="/api/community", tags=["community"])


def _resolve_working_set(payload: CommunityRequest) -> None:
    """Resolve scope_mode='working_set' into scope_mode='ids' + source_ids, in place."""
    if payload.scope_mode != "working_set":
        return
    if not payload.working_set_id:
        raise HTTPException(status_code=422, detail="working_set_id is required when scope_mode is 'working_set'")
    with get_connection() as conn:
        row = conn.execute(
            "SELECT source_ids FROM working_sets WHERE id = %s",
            (payload.working_set_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"working set not found: {payload.working_set_id}")
    source_ids_raw = row[0]
    if isinstance(source_ids_raw, str):
        source_ids_raw = json.loads(source_ids_raw)
    payload.scope_mode = "ids"
    payload.source_ids = list(source_ids_raw) if isinstance(source_ids_raw, list) else []


def _build_params(payload: CommunityRequest) -> dict:
    ro = payload.retrieve_options
    return {
        "scope_mode": payload.scope_mode,
        "source_ids": payload.source_ids,
        "criteria": payload.criteria,
        "filters": payload.filters,
        "search_options": {
            "limit": payload.search_options.limit,
            "min_score": payload.search_options.min_score,
        },
        "retrieve_options": {
            "seed_count": ro.seed_count,
            "result_count": ro.result_count,
            "rrf_k": ro.rrf_k,
            "entity_confidence_threshold": ro.entity_confidence_threshold,
            "first_hop_similarity_threshold": ro.first_hop_similarity_threshold,
            "second_hop_similarity_threshold": ro.second_hop_similarity_threshold,
            "trace": ro.trace,
        },
        "semantic_threshold": payload.community_options.semantic_threshold,
        "cutoff": payload.community_options.cutoff,
        "min_community_size": payload.community_options.min_community_size,
        "top_k_chunks": payload.community_options.top_k_chunks,
        "summarize_model": payload.summarize_model,
        "cross_source_top_k": payload.community_options.cross_source_top_k,
        "max_cross_source_queries": payload.community_options.max_cross_source_queries,
        "source_cooc_weight": payload.community_options.source_cooc_weight,
        "resolution": payload.community_options.resolution,
    }


@router.post("")
def community_sync(payload: CommunityRequest) -> dict:
    _resolve_working_set(payload)
    try:
        result = detect_communities(
            scope_mode=payload.scope_mode,
            source_ids=payload.source_ids,
            criteria=payload.criteria,
            filters=payload.filters,
            search_options=_build_params(payload)["search_options"],
            retrieve_options=_build_params(payload)["retrieve_options"],
            semantic_threshold=payload.community_options.semantic_threshold,
            cutoff=payload.community_options.cutoff,
            min_community_size=payload.community_options.min_community_size,
            top_k_chunks=payload.community_options.top_k_chunks,
            summarize_model=payload.summarize_model,
            cross_source_top_k=payload.community_options.cross_source_top_k,
            max_cross_source_queries=payload.community_options.max_cross_source_queries,
            source_cooc_weight=payload.community_options.source_cooc_weight,
            resolution=payload.community_options.resolution,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        params = _build_params(payload)
        resolved_source_ids = payload.source_ids if payload.scope_mode == "ids" else result.get("metadata", {}).get("source_ids", [])
        run_id = create_run(params, resolved_source_ids)
        complete_run(run_id, result, resolved_source_ids)
        result["run_id"] = run_id
    except Exception:
        pass

    return result


@router.post("/runs")
def start_run(payload: CommunityRequest) -> dict:
    _resolve_working_set(payload)
    params = _build_params(payload)
    try:
        run_id = create_run(params, payload.source_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e

    thread = threading.Thread(target=execute_run, args=(run_id,), daemon=True)
    thread.start()

    return {"run_id": run_id}


@router.get("/runs")
def list_community_runs_api(
    limit: int = Query(default=20, gt=0, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    return list_runs(limit=limit, offset=offset)


@router.get("/runs/{run_id}")
def get_community_run(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@router.get("/runs/{run_id}/events")
def run_events(run_id: str):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    def event_iter():
        seen = 0
        for event in stream_run_events(run_id, seen):
            # Recompute seen count from stage_log length
            if event["event"] == "stage":
                try:
                    data = json.loads(event["data"])
                    if isinstance(data, dict) and "stage" in data:
                        seen = seen + 1
                except (json.JSONDecodeError, KeyError):
                    pass
            yield event

    # sse-starlette defaults to "\r\n" line separators; the frontend parser
    # (shared with the hand-rolled answer stream) splits on "\n\n", so the
    # separator must match here or events never get parsed out of the buffer.
    return EventSourceResponse(event_iter(), sep="\n")
