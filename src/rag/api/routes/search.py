import requests.exceptions
from fastapi import APIRouter, HTTPException

from rag.api.schemas import SearchRequest, SearchResponse, SearchResult
from rag.retrieval import hybrid_search


router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    try:
        results = hybrid_search(payload.query, limit=payload.limit, min_score=payload.min_score)
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=503, detail="Embedding service unavailable") from exc
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
