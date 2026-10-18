# Ultralytics YOLO Code Assistant

[![Tests](https://github.com/talipturkmen/ultratics-yolo-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/talipturkmen/ultratics-yolo-assistant/actions/workflows/tests.yml)

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
   - `INDEX_SUBDIRS` (comma-separated list of subdirectories to index)
   - `ULTRALYTICS_PATH` (path to the cloned Ultralytics YOLO code)
2. Make sure you have the `uv` tool installed (Rust-based Python manager). If not, install via pip: `pip install uv`.
3. (Optional) Edit `prompts/system_prompt.txt` to customize the assistant behavior.

### Create the Atlas Vector Search Index

Create an Atlas Search index (Vector Search) on your embeddings collection. The index must map `embedding` as a 384-dim vector (for the default embedding model).

Example index definition (name it `vector_index`, or set `MONGODB_VECTOR_INDEX` accordingly):

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

Then open `http://localhost:8814` in your browser.

The app binds to `127.0.0.1` by default. To expose it on your network, set `GRADIO_HOST=0.0.0.0` (and `GRADIO_PORT` to change the port). Anyone who can reach it can spend your OpenRouter credits.

### Tests

Run the unit tests:

```bash
uv run python -m unittest discover -s tests
```

## Design Notes

- **Chunking strategy**: Parse each Python file with AST and chunk by top-level definitions (classes/functions) plus any global code gaps. This keeps chunks semantically coherent while preserving module-level context. Oversized chunks are further split by token count with overlap to fit embedding limits. See `src/indexer.py`.
- **Metadata extracted**: Each chunk stores `path`, `symbol` (class/function name when available), `start_line`, `end_line`, `language`, `chunk_id`, and `updated_at`. This supports precise citations in responses. See `src/indexer.py`.
- **Embedding model choice**: Default is `sentence-transformers/all-MiniLM-L6-v2` (384 dims). It is fast, lightweight, and aligns with the 384-dim Atlas vector index example. The embedding model is configurable via `EMBED_MODEL`. See `src/settings.py`.
- **LLM choice**: Default is `deepseek/deepseek-r1:free` on OpenRouter to keep usage free and simple. The model is configurable via `OPENROUTER_MODEL`. See `src/settings.py`.
- **Trade-offs**: The indexer clears and rebuilds the collection on each run for simplicity. This avoids sync complexity but is slower for large repos. The AST-based chunking is deterministic but may split logic across files or miss cross-file context.
- For detailed design notes please see `DESIGN.md`.

## Future Work

- [ ] Incremental indexing keyed by file hash to avoid full re-indexing on each run.
- [ ] Smarter chunking that uses docstrings or semantic boundaries beyond top-level AST nodes.
- [ ] Multi-repo or versioned indexing to compare behavior across Ultralytics releases.
- [ ] **Scale with separate services**: Isolate indexing, retrieval, and serving to avoid contention and allow independent scaling of CPU/GPU and DB workloads.
- [ ] **Queue indexing jobs**: Use a job queue to smooth spikes, retry failures, and keep ingestion resilient during large repo updates.
- [ ] **Cache embeddings**: Reuse embeddings for unchanged files to cut compute cost and reduce indexing latency.
- [ ] **Versioned indexes**: Keep per-release indexes to support historical queries and safe rollbacks.
- [ ] **Observability**: Add metrics, tracing, and alerts to detect slow queries, drift, or embedding/model failures early.
- [ ] **LLM gateway**: Rate-limit and retry LLM calls to control spend and improve reliability under load.
- [ ] Reranking and answer validation for higher precision on ambiguous questions.
- [ ] Add evaluation scripts and a small test set of Q&A pairs for regression checks.

## Design Document

See `DESIGN.md`.

## Demo Screenshot

Please see the images unders `examples` folder
