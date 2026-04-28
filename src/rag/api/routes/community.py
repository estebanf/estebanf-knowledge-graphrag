from fastapi import APIRouter

from rag.api.schemas import CommunityRequest
from rag.community import detect_communities

router = APIRouter(prefix="/api/community", tags=["community"])


@router.post("")
def community_route(payload: CommunityRequest) -> dict:
    ro = payload.retrieve_options
    return detect_communities(
        scope_mode=payload.scope_mode,
        source_ids=payload.source_ids,
        criteria=payload.criteria,
        filters=payload.filters,
        search_options={
            "limit": payload.search_options.limit,
            "min_score": payload.search_options.min_score,
        },
        retrieve_options={
            "seed_count": ro.seed_count,
            "result_count": ro.result_count,
            "rrf_k": ro.rrf_k,
            "entity_confidence_threshold": ro.entity_confidence_threshold,
            "first_hop_similarity_threshold": ro.first_hop_similarity_threshold,
            "second_hop_similarity_threshold": ro.second_hop_similarity_threshold,
            "trace": ro.trace,
        },
        semantic_threshold=payload.community_options.semantic_threshold,
        cutoff=payload.community_options.cutoff,
        min_community_size=payload.community_options.min_community_size,
        top_k_chunks=payload.community_options.top_k_chunks,
        summarize_model=payload.summarize_model,
        cross_source_top_k=payload.community_options.cross_source_top_k,
        max_cross_source_queries=payload.community_options.max_cross_source_queries,
    )
