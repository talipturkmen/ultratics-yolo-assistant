# Design Document: Ultralytics YOLO Code Assistant

## Goals
Answer questions about the Ultralytics YOLO codebase with citations while keeping the system simple, reproducible, and dependency-light, and avoiding pre-built RAG frameworks.

## Architecture Overview
The system has three stages:
1. **Indexing**: Parse and chunk YOLO source files, embed chunks, and store them in MongoDB Atlas.
2. **Retrieval**: Run vector search in Atlas to fetch the most relevant chunks.
3. **Answering**: Build a system+user prompt from the retrieved chunks and call OpenRouter to generate a response with citations.

## Components
- `src/indexer.py`: Walks the repo, chunks code using AST boundaries, embeds, and writes to MongoDB.
- `src/retriever.py`: Executes Atlas Vector Search queries and returns top-k chunks.
- `src/llm.py`: Builds prompt and calls OpenRouter for a final answer.
- `src/app.py`: Gradio UI for interactive Q&A.
- `src/settings.py`: Configuration via `.env` using Pydantic settings.
- `scripts/index.py`: CLI entrypoint to run indexing.

## Data Flow
1. **Indexing**
   - Input: `ULTRALYTICS_PATH`, `INDEX_SUBDIRS`, chunking settings, embedding model.
   - For each `.py` file: parse AST, create chunks per top-level node + global gaps.
   - Large chunks are split by token size with overlap.
   - Store each chunk with embeddings and metadata (`path`, `symbol`, `start_line`, `end_line`, etc.).

2. **Query**
   - User enters a question in Gradio.
   - `retriever.py` converts the question to an embedding and runs vector search in Atlas.
   - Top-k chunks are passed into the LLM prompt.

3. **Answer**
   - `llm.py` builds a system prompt + user prompt.
   - OpenRouter generates an answer with required sections and citations.

## Chunking Strategy
- **Why AST**: Chunks follow function/class boundaries so each chunk is semantically coherent and easier to cite.
- **Global code**: Top-level statements between definitions are captured as their own chunks to preserve module setup and constants.
- **Token splitting**: Oversized chunks are split by token count with overlap so they fit model limits while keeping some local context.

## Metadata and Citations
Stored metadata fields enable precise citations in the final answer:
- `path`, `start_line`, `end_line`, `symbol`, `language`, `chunk_id`, `updated_at`.
LLM output is instructed to cite file paths and line numbers.

## Model Choices
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (fast, free, 384 dims), chosen for hardware-limited environments.
- **LLM**: Default OpenRouter free model `deepseek/deepseek-r1:free`.
Both are configurable via `.env`.

## Trade-offs
- Full re-indexing is performed on each run (simple, but slower).
- No reranking beyond vector search is used (fast, but nuances can be missed).
- A general-purpose embedding model is used instead of a code-specialized model due to hardware limitations (fast and free), but context is shorter (512) and code recall can be lower than larger paid models (e.g., OpenAI text-embedding-ada); a stronger alternative could be jina-v2.

## Future Improvements
- [ ] **Incremental indexing with file hashes**: Track file hashes and only re-embed changed files to cut re-indexing time and costs.
- [ ] **Reranking (e.g., cross-encoder)**: Re-score top results with a stronger model to improve precision on close or ambiguous matches.
- [ ] **Larger eval set and regression tests**: Build a small gold Q&A set to measure retrieval/answer quality and prevent regressions over time.
- [ ] **Multi-repo support and versioned indexes**: Keep separate indexes per repo/version to handle multiple codebases and allow historical comparisons.
- [ ] **Scale with separate services**: Isolate indexing, retrieval, and serving to prevent resource contention and enable independent scaling.
- [ ] **Queue indexing jobs**: Smooth traffic spikes, coordinate retries, and keep ingestion durable during large updates.
- [ ] **Cache embeddings**: Reuse embeddings for unchanged files to reduce compute cost and speed up re-indexing.
- [ ] **Shard/replicate vector store**: Improve throughput and availability as the index grows.
- [ ] **Observability**: Add metrics, tracing, and alerts to detect latency regressions and failures early.
- [ ] **LLM gateway**: Apply rate limits and retries to control cost and maintain reliability under load.
