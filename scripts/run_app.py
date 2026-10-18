import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a script file:
#   uv run scripts/run_app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import app, retriever
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
    # Ensure the LLM module sees OpenRouter settings even if `.env` isn't loaded into the process env.
    os.environ.setdefault("OPENROUTER_API_KEY", config.OPENROUTER_API_KEY)
    os.environ.setdefault("OPENROUTER_MODEL", config.OPENROUTER_MODEL)
    # Initialize the retriever (connect to DB and load embed model)
    retriever.init_retriever(
        mongo_uri=config.MONGODB_URI,
        db_name=config.MONGODB_DB,
        collection_name=config.MONGODB_COLLECTION,
        embed_model_name=config.EMBED_MODEL,
        vector_index_name=config.MONGODB_VECTOR_INDEX,
        embedding_path=config.MONGODB_EMBEDDING_PATH,
    )
    # Launch the Gradio app
    interface = app.create_interface()
    server_name = os.getenv("GRADIO_HOST", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_PORT", "8814"))
    interface.launch(server_name=server_name, server_port=server_port)
    
if __name__ == "__main__":
    main()
