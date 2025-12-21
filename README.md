# Ultralytics YOLO Code Assistant

This project is a small Retrieval-Augmented Generation (RAG) based code assistant for the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) repository. It can answer questions about the YOLO codebase by retrieving relevant code snippets and using an LLM (via OpenRouter) to generate helpful answers with citations.

## Features

- **Code Indexing**: Indexes the YOLO code (models, engine, data directories) into a MongoDB Atlas vector store.
- **Semantic Search**: Retrieves relevant code chunks using MongoDB Atlas Vector Search.
- **LLM Answers**: Uses an LLM (via OpenRouter) to generate answers that explain what the code does and how to use it, with file path and line number citations.
- **User Interface**: Provides an interactive Q&A interface built with Gradio.

## Setup

### Prerequisites

- **Ultralytics YOLO Source**: Clone the Ultralytics YOLO repository (e.g., `git clone https://github.com/ultralytics/ultralytics.git`) and note the path. This will be used to index the code.
- **MongoDB Atlas**: Create a free MongoDB Atlas cluster. Enable **Atlas Vector Search** on your cluster (available on M0 clusters and above). Get your connection URI and create a database and collection for the embeddings.
- **OpenRouter API Key**: Sign up at [OpenRouter](https://openrouter.ai) to get a free API key. The default model used is `deepseek/deepseek-r1:free`, which is a free model on OpenRouter.

### Configuration

1. Copy `.env.example` to `.env` and fill in all the required values:
   - `MONGODB_URI`, `MONGODB_DB`, `MONGODB_COLLECTION`
   - `MONGODB_VECTOR_INDEX` (Atlas Search index name, default `vector_index`)
   - `MONGODB_EMBEDDING_PATH` (field name for vectors, default `embedding`)
   - `OPENROUTER_API_KEY` and optionally `OPENROUTER_MODEL`
   - `EMBED_MODEL` (default is a small 384-dimensional model for embeddings)
   - `CHUNK_TOKEN_TARGET` and `CHUNK_OVERLAP_TOKENS` (chunk size and overlap settings)
   - `ULTRALYTICS_PATH` (path to the cloned Ultralytics YOLO code)
2. Make sure you have the `uv` tool installed (Rust-based Python manager). If not, install via pip: `pip install uv`.

### Create the Atlas Vector Search Index

Create an Atlas Search index (Vector Search) on your embeddings collection. The index must map `embedding` as a 384-dim vector (for the default embedding model).

Example index definition (name it `default`, or set `MONGODB_VECTOR_INDEX` accordingly):

```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine" }
  ]
}
```

### Index the Codebase

Before asking questions, you need to index the YOLO code:

```bash
uv sync                # installs the dependencies in a virtual environment
uv run scripts/index.py  # run the indexing script
```

### Run the App

After indexing completes, start the Gradio UI:

```bash
uv run scripts/run_app.py
```

Then open `http://localhost:7860` in your browser.
