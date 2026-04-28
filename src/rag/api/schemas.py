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


class SearchOptions(BaseModel):
    limit: int = Field(default=10, gt=0)
    min_score: float = Field(default=0.0, ge=0.0)


class RetrieveOptions(BaseModel):
    seed_count: Optional[int] = Field(default=None, gt=0)
    result_count: Optional[int] = Field(default=None, gt=0)
    rrf_k: Optional[int] = Field(default=None, gt=0)
    entity_confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    first_hop_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    second_hop_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    trace: bool = False


class CommunityOptions(BaseModel):
    semantic_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cutoff: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_community_size: Optional[int] = Field(default=None, gt=0)
    top_k_chunks: Optional[int] = Field(default=None, gt=0)
    cross_source_top_k: Optional[int] = Field(default=None, gt=0)
    max_cross_source_queries: Optional[int] = Field(default=None, gt=0)


class CommunityRequest(BaseModel):
    scope_mode: str = Field(..., pattern="^(ids|search|retrieve)$")
    source_ids: list[str] = Field(default_factory=list)
    criteria: list[str] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)
    search_options: SearchOptions = Field(default_factory=SearchOptions)
    retrieve_options: RetrieveOptions = Field(default_factory=RetrieveOptions)
    community_options: CommunityOptions = Field(default_factory=CommunityOptions)
    summarize_model: Optional[str] = None


class SourceDetail(BaseModel):
    source_id: str
    name: Optional[str] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    storage_path: str
    metadata: dict[str, Any]
    markdown_content: str
