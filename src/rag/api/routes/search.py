from fastapi import APIRouter

from rag.api.schemas import SearchRequest, SearchResponse, SearchResult
from rag.retrieval import hybrid_search


router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    results = hybrid_search(payload.query, limit=payload.limit, min_score=payload.min_score)
    return SearchResponse(
        results=[
            SearchResult(
                score=result.score,
                chunk=result.chunk,
                chunk_id=result.chunk_id,
                source_id=result.source_id,
                source_path=result.source_path,
                source_metadata=result.source_metadata,
            )
            for result in results
        ]
    )
