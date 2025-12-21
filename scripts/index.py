import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a script file:
#   uv run scripts/index.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import indexer
from src.settings import Settings
from pydantic import ValidationError

def main():
    logging.basicConfig(level=logging.INFO)
    try:
        config = Settings()
    except ValidationError as e:
        print("Configuration error:", e)
        print("Please ensure all required environment variables are set (see .env.example).")
        return
    # Run the indexing process
    indexer.index_codebase(
        repo_path=config.ULTRALYTICS_PATH,
        db_uri=config.MONGODB_URI,
        db_name=config.MONGODB_DB,
        collection_name=config.MONGODB_COLLECTION,
        embed_model_name=config.EMBED_MODEL,
        chunk_token_target=config.CHUNK_TOKEN_TARGET,
        chunk_overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
        subdirs=config.index_subdirs_list(),
    )

if __name__ == "__main__":
    main()
