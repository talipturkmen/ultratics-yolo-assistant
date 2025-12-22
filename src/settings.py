import json
from pathlib import Path
import logging
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
    INDEX_SUBDIRS: str = "ultralytics/models,ultralytics/engine,ultralytics/data"
    ULTRALYTICS_PATH: str

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[1] / ".env"),
        env_file_encoding="utf-8",
    )
    # use loggging to indicate settings loaded

    logging.info("Settings loaded from .env file")
    logging.debug("Settings config: %s", model_config)

    def index_subdirs_list(self) -> list[str]:
        value = self.INDEX_SUBDIRS
        if not value:
            return []
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in value.split(",") if item.strip()]
