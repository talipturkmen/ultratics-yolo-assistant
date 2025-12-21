from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MONGODB_URI: str
    MONGODB_DB: str
    MONGODB_COLLECTION: str
    MONGODB_VECTOR_INDEX: str = "vector_index"
    MONGODB_EMBEDDING_PATH: str = "embedding"
    OPENROUTER_API_KEY: str
    OPENROUTER_MODEL: str = "deepseek/deepseek-r1:free"
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_TOKEN_TARGET: int = 500
    CHUNK_OVERLAP_TOKENS: int = 50
    ULTRALYTICS_PATH: str

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[1] / ".env"),
        env_file_encoding="utf-8",
    )
    print("Settings loaded from .env file")
    print("Settings:", model_config)
