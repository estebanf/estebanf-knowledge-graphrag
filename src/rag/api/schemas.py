from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class DescribeImageRequest(BaseModel):
    image_base64: str = Field(..., min_length=1, description="Base64-encoded image data to describe")
    mime_type: str = Field(..., description="MIME type of the image (e.g. image/png, image/jpeg)")


class DescribeImageResponse(BaseModel):
    description: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    limit: int = Field(default=10, gt=0, description="Maximum number of results per type (chunks and insights)")
    min_score: float = Field(default=0.7, ge=0.0, description="Minimum similarity score threshold for results")


class SearchResult(BaseModel):
    score: float
    chunk: str
    chunk_id: str
    source_id: str
    source_path: str
    source_metadata: dict[str, Any]


class InsightSourceInfo(BaseModel):
    source_id: str
    source_path: str
    source_metadata: dict[str, Any]


class InsightResult(BaseModel):
    score: float
    insight: str
    insight_id: str
    topics: list[str]
    sources: list[InsightSourceInfo]


class SearchResults(BaseModel):
    chunks: list[SearchResult]
    insights: list[InsightResult]


class SearchResponse(BaseModel):
    results: SearchResults


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    source_ids: list[str] = Field(default_factory=list, description="Optional source IDs to restrict retrieval scope")
    filters: dict[str, str] = Field(default_factory=dict, description="Key-value metadata filters applied to retrieval")
    seed_count: Optional[int] = Field(default=None, gt=0, description="Maximum number of seed chunks used for graph expansion")
    result_count: Optional[int] = Field(default=None, gt=0, description="Maximum number of final results to return")
    rrf_k: Optional[int] = Field(default=None, gt=0, description="Reciprocal Rank Fusion parameter; higher values give more weight to dense results")
    entity_confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum confidence threshold for including entities in graph expansion")
    first_hop_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Maximum similarity distance for first-hop entity expansion")
    second_hop_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Maximum similarity distance for second-hop entity expansion")
    trace: bool = Field(default=False, description="Whether to include detailed retrieval trace in the response")


class AnswerRequest(RetrieveRequest):
    model: str = Field(..., description="LLM model identifier used for answer generation")


class SearchOptions(BaseModel):
    limit: int = Field(default=10, gt=0, description="Maximum number of results per type (chunks and insights)")
    min_score: float = Field(default=0.0, ge=0.0, description="Minimum similarity score threshold for results")


class RetrieveOptions(BaseModel):
    seed_count: Optional[int] = Field(default=None, gt=0, description="Maximum number of seed chunks used for graph expansion")
    result_count: Optional[int] = Field(default=None, gt=0, description="Maximum number of final results to return")
    rrf_k: Optional[int] = Field(default=None, gt=0, description="Reciprocal Rank Fusion parameter; higher values give more weight to dense results")
    entity_confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum confidence threshold for including entities in graph expansion")
    first_hop_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Maximum similarity distance for first-hop entity expansion")
    second_hop_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Maximum similarity distance for second-hop entity expansion")
    trace: bool = Field(default=False, description="Whether to include detailed retrieval trace in the response")


class CommunityOptions(BaseModel):
    semantic_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Cosine similarity threshold for connecting entities within a community")
    cutoff: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum chunk relevance score for inclusion in a community")
    min_community_size: Optional[int] = Field(default=None, gt=0, description="Minimum number of entities required to form a community")
    top_k_chunks: Optional[int] = Field(default=None, gt=0, description="Maximum number of chunks to surface per community")
    cross_source_top_k: Optional[int] = Field(default=None, gt=0, description="Maximum cross-source ANN neighbors fetched per entity during community building")
    max_cross_source_queries: Optional[int] = Field(default=None, gt=0, description="Hard cap on total cross-source ANN queries during community building")
    source_cooc_weight: Optional[float] = Field(default=None, ge=0.0, description="Extra weight added when entities co-occur in the same source during community detection")
    resolution: Optional[float] = Field(default=None, gt=0.0, description="Community granularity. 1.0 is balanced; below 1.0 yields fewer, larger communities; above 1.0 yields more, smaller ones")


class CommunityRequest(BaseModel):
    scope_mode: str = Field(..., pattern="^(ids|search|retrieve|working_set)$", description="How to scope the community: by explicit source IDs, search query, retrieval query, or a saved working set")
    source_ids: list[str] = Field(default_factory=list, description="Source IDs when scope_mode is 'ids'")
    working_set_id: Optional[str] = Field(default=None, description="Working set ID when scope_mode is 'working_set'")
    criteria: list[str] = Field(default_factory=list, description="Retrieval criteria when scope_mode is 'retrieve'")
    filters: dict[str, str] = Field(default_factory=dict, description="Key-value metadata filters applied to the scope")
    search_options: SearchOptions = Field(default_factory=SearchOptions, description="Search tuning parameters when scope_mode is 'search'")
    retrieve_options: RetrieveOptions = Field(default_factory=RetrieveOptions, description="Retrieval tuning parameters when scope_mode is 'retrieve'")
    community_options: CommunityOptions = Field(default_factory=CommunityOptions, description="Community detection tuning parameters")
    summarize_model: Optional[str] = Field(default=None, description="Optional LLM model to generate summaries for each detected community")


class SourceDetail(BaseModel):
    source_id: str
    name: Optional[str] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    storage_path: str
    metadata: dict[str, Any]
    markdown_content: str


class SourceSummary(BaseModel):
    source_id: str
    name: Optional[str] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    metadata: dict[str, Any]
    created_at: datetime
    insight_count: int = 0


class SourceListResponse(BaseModel):
    sources: list[SourceSummary]
    total: int
    limit: int
    offset: int


class SourceInsight(BaseModel):
    insight_id: str
    insight: str
    topics: list[str]
    chunk_id: str
    chunk_index: Optional[int] = None
    chunk_preview: str


class SourceInsightsResponse(BaseModel):
    insights: list[SourceInsight]
