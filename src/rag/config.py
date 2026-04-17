from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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

    # Embedding
    EMBEDDING_DIMENSIONS: Annotated[int, Field(gt=0)] = 4096

    # Chunking / validation
    CHUNK_VALIDATION_SAMPLE_RATE: Annotated[float, Field(ge=0.0, le=1.0)] = 0.10
    CHUNK_VALIDATION_SAMPLE_RATE_HIGH_STAKES: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25


settings = Settings()
