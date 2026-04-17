from pathlib import Path
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
    MODEL_METADATA_EXTRACTION: str = "google/gemma-3-4b-it"


settings = Settings()
