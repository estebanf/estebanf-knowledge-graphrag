import requests.exceptions
from fastapi import APIRouter, HTTPException

from rag.api.schemas import InsightResult, InsightSourceInfo, SearchRequest, SearchResponse, SearchResult, SearchResults
from rag.retrieval import hybrid_search


router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    try:
        results = hybrid_search(payload.query, limit=payload.limit, min_score=payload.min_score)
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=503, detail="Embedding service unavailable") from exc
    return SearchResponse(
        results=SearchResults(
            chunks=[
                SearchResult(
                    score=r.score,
                    chunk=r.chunk,
                    chunk_id=r.chunk_id,
                    source_id=r.source_id,
                    source_path=r.source_path,
                    source_metadata=r.source_metadata,
                )
                for r in results.chunks
            ],
            insights=[
                InsightResult(
                    score=r.score,
                    insight=r.insight,
                    insight_id=r.insight_id,
                    topics=r.topics,
                    sources=[
                        InsightSourceInfo(
                            source_id=s.source_id,
                            source_path=s.source_path,
                            source_metadata=s.source_metadata,
                        )
                        for s in r.sources
                    ],
                )
                for r in results.insights
            ],
        )
    )
