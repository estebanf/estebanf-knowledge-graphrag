from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, gt=0)
    min_score: float = Field(default=0.7, ge=0.0)


class SearchResult(BaseModel):
    score: float
    chunk: str
    chunk_id: str
    source_id: str
    source_path: str
    source_metadata: dict[str, Any]


class SearchResponse(BaseModel):
    results: list[SearchResult]


class RetrieveRequest(BaseModel):
    query: str
    source_ids: list[str] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)
    seed_count: Optional[int] = Field(default=None, gt=0)
    result_count: Optional[int] = Field(default=None, gt=0)
    rrf_k: Optional[int] = Field(default=None, gt=0)
    entity_confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    first_hop_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    second_hop_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    trace: bool = False


class AnswerRequest(RetrieveRequest):
    model: str


class SourceDetail(BaseModel):
    source_id: str
    name: Optional[str] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    storage_path: str
    metadata: dict[str, Any]
    markdown_content: str
