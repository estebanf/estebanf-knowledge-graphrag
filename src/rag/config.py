from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    POSTGRES_URL: str = "postgresql://rag:changeme@localhost:5432/rag"
    STORAGE_BASE_PATH: Path = Path("./data/documents")
    LOG_LEVEL: str = "INFO"
    OPENROUTER_API_KEY: str = ""

    # Model roles
    MODEL_METADATA_EXTRACTION: str = "google/gemma-3-4b-it"
    MODEL_DOC_PROFILING: str = "google/gemma-3-4b-it"
    MODEL_CHUNK_VALIDATION: str = "qwen/qwen2.5-7b-instruct"
    MODEL_PROPOSITION_CHUNKING: str = "qwen/qwen2.5-14b-instruct"
    MODEL_EMBEDDING: str = "qwen/qwen3-embedding-8b"
    MODEL_RETRIEVAL_QUERY_VARIANTS: str = "google/gemini-2.5-flash-lite"
    MODEL_RETRIEVAL_GRAPH: str = "google/gemini-2.5-flash-lite"
    MODEL_RETRIEVAL_RERANKER: str = "cohere/rerank-v3.5"

    # Embedding
    EMBEDDING_DIMENSIONS: Annotated[int, Field(gt=0)] = 4096

    # Chunking / validation
    CHUNK_VALIDATION_SAMPLE_RATE: Annotated[float, Field(ge=0.0, le=1.0)] = 0.10
    CHUNK_VALIDATION_SAMPLE_RATE_HIGH_STAKES: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25

    # Graph pipeline
    MEMGRAPH_URL: str = "bolt://localhost:7687"
    MODEL_ENTITY_EXTRACTION: str = "qwen/qwen-2.5-7b-instruct"
    MODEL_RELATIONSHIP_EXTRACTION: str = "qwen/qwen-2.5-7b-instruct"
    MODEL_IMAGE_DESCRIPTION: str = "google/gemini-2.0-flash-lite-001"
    RELATIONSHIP_CONFIDENCE_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75
    ENTITY_DEDUP_COSINE_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] = 0.92

    # Retrieval
    RETRIEVAL_RRF_K: Annotated[int, Field(gt=0)] = 60
    RETRIEVAL_RRF_SCORE_FLOOR: float = 0.0
    RETRIEVAL_SEED_COUNT: Annotated[int, Field(gt=0)] = 10
    RETRIEVAL_RESULT_COUNT: Annotated[int, Field(gt=0)] = 5
    RETRIEVAL_MAX_DECOMPOSED_QUERIES: Annotated[int, Field(gt=0)] = 5
    RETRIEVAL_FIRST_STAGE_TOP_N: Annotated[int, Field(gt=0)] = 20
    RETRIEVAL_FUSED_CANDIDATE_COUNT: Annotated[int, Field(gt=0)] = 50
    RETRIEVAL_ENTITY_SELECTION_COUNT: Annotated[int, Field(gt=0)] = 5
    RETRIEVAL_SECOND_HOP_SELECTION_COUNT: Annotated[int, Field(gt=0)] = 5
    RETRIEVAL_FIRST_HOP_CHUNK_COUNT: Annotated[int, Field(gt=0)] = 5
    RETRIEVAL_SECOND_HOP_CHUNK_COUNT: Annotated[int, Field(gt=0)] = 5
    RETRIEVAL_FIRST_HOP_SIMILARITY_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    RETRIEVAL_SECOND_HOP_SIMILARITY_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    RETRIEVAL_ENTITY_CONFIDENCE_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75
    RETRIEVAL_MAX_GRAPH_LLM_CALLS: Annotated[int, Field(gt=0)] = 100
    RETRIEVAL_MAX_GRAPH_EXPANSION_MS: Annotated[int, Field(gt=0)] = 4000
    RETRIEVAL_MAX_GRAPH_EXPANSION_MS_PER_SEED: Annotated[int, Field(gt=0)] = 4000
    RETRIEVAL_TEXT_SEARCH_CONFIG: str = "english"
    RETRIEVAL_WEIGHT_ORIGINAL: Annotated[float, Field(ge=0.0)] = 1.0
    RETRIEVAL_WEIGHT_DECOMPOSED: Annotated[float, Field(ge=0.0)] = 1.0
    RETRIEVAL_WEIGHT_EXPANDED: Annotated[float, Field(ge=0.0)] = 0.85
    RETRIEVAL_WEIGHT_STEP_BACK: Annotated[float, Field(ge=0.0)] = 0.75
    RETRIEVAL_WEIGHT_HYDE: Annotated[float, Field(ge=0.0)] = 0.65
    RETRIEVAL_FINAL_ROOT_WEIGHT: Annotated[float, Field(ge=0.0)] = 0.60
    RETRIEVAL_FINAL_FIRST_HOP_WEIGHT: Annotated[float, Field(ge=0.0)] = 0.25
    RETRIEVAL_FINAL_SECOND_HOP_WEIGHT: Annotated[float, Field(ge=0.0)] = 0.15
    RETRIEVAL_MULTI_PATH_BONUS: Annotated[float, Field(ge=0.0)] = 0.05
    RETRIEVAL_SAME_SOURCE_NEIGHBOR_WINDOW: Annotated[int, Field(ge=0)] = 2
    RETRIEVAL_SAME_SOURCE_NEIGHBOR_COUNT: Annotated[int, Field(gt=0)] = 3
    RETRIEVAL_EXPANSION_MIN_TOKENS: Annotated[int, Field(gt=0)] = 200
    RETRIEVAL_EXPANSION_MAX_TOKENS: Annotated[int, Field(gt=0)] = 600
    RETRIEVAL_TRACE_MAX_CANDIDATES: Annotated[int, Field(gt=0)] = 5
    RETRIEVAL_TRACE_MAX_ENTITIES: Annotated[int, Field(gt=0)] = 5

    # Search command
    SEARCH_DEFAULT_LIMIT: Annotated[int, Field(gt=0)] = 10
    SEARCH_MIN_SCORE: Annotated[float, Field(ge=0.0)] = 0.7

    # Community summarization
    COMMUNITY_SEMANTIC_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] = 0.85
    COMMUNITY_SOURCE_COOC_WEIGHT: Annotated[float, Field(ge=0.0)] = 0.1
    COMMUNITY_CUTOFF: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    COMMUNITY_MIN_COMMUNITY_SIZE: Annotated[int, Field(gt=0)] = 3
    COMMUNITY_TOP_K_CHUNKS: Annotated[int, Field(gt=0)] = 5
    COMMUNITY_SUMMARIZATION_PROMPT: str = ""

    # Worker
    WORKER_POLL_INTERVAL: int = 5       # seconds between polls when queue is empty
    WORKER_STUCK_JOB_MINUTES: int = 30  # minutes before a processing job is declared stuck

    def retrieval_variant_weight(self, variant_name: str) -> float:
        if variant_name.startswith("decomposed_"):
            return self.RETRIEVAL_WEIGHT_DECOMPOSED
        return {
            "original": self.RETRIEVAL_WEIGHT_ORIGINAL,
            "expanded": self.RETRIEVAL_WEIGHT_EXPANDED,
            "step_back": self.RETRIEVAL_WEIGHT_STEP_BACK,
            "hyde": self.RETRIEVAL_WEIGHT_HYDE,
        }.get(variant_name, 1.0)


settings = Settings()
