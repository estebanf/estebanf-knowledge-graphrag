from fastapi import APIRouter

from rag.api.schemas import RetrieveRequest
from rag.retrieval import retrieve


router = APIRouter(prefix="/api/retrieve", tags=["retrieve"])


@router.post("")
def retrieve_route(payload: RetrieveRequest) -> dict:
    return retrieve(
        query=payload.query,
        source_ids=payload.source_ids,
        filters=payload.filters,
        seed_count=payload.seed_count,
        result_count=payload.result_count,
        rrf_k=payload.rrf_k,
        entity_confidence_threshold=payload.entity_confidence_threshold,
        first_hop_similarity_threshold=payload.first_hop_similarity_threshold,
        second_hop_similarity_threshold=payload.second_hop_similarity_threshold,
        trace=payload.trace,
        trace_printer=None,
    )
