import json
import threading

from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from rag.api.schemas import CommunityRequest
from rag.community import detect_communities
from rag.community_runs import (
    create_run,
    execute_run,
    get_run,
    list_runs,
    stream_run_events,
)

router = APIRouter(prefix="/api/community", tags=["community"])


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
        run_id = create_run(params, payload.source_ids if payload.scope_mode == "ids" else result.get("metadata", {}).get("source_ids", []))
        result["run_id"] = run_id
    except Exception:
        pass

    return result


@router.post("/runs")
def start_run(payload: CommunityRequest) -> dict:
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

    return EventSourceResponse(event_iter())
