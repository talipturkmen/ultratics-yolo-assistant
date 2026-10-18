import re
from typing import Optional
from pymongo import MongoClient
import logging

# Globals for connection and embedding model
_client = None
_collection = None
_embedder = None
_embed_model_name = None
_vector_index_name = "vector_index"
_embedding_path = "embedding"

def _mongo_connection_hint(error: Exception) -> Optional[str]:
    message = str(error)
    if "SSL handshake failed" in message or "TLSV1_ALERT_INTERNAL_ERROR" in message:
        return (
            "MongoDB TLS handshake failed. Re-copy the official Atlas connection string (Connect → Drivers) "
            "and ensure your current IP is allowed in Atlas (Network Access)."
        )
    return None

def _redact_mongo_uri(uri: str) -> str:
    return re.sub(r"(?<=://)([^:@/]+):([^@/]+)@", r"\1:***@", uri)

def _try_log_collection_diagnostics():
    if _collection is None:
        return

    try:
        count = _collection.estimated_document_count()
        logging.info(f"Mongo collection '{_collection.full_name}' doc count (estimated): {count}")
        sample = _collection.find_one({}, projection={"_id": 1, "meta.path": 1, "embedding": 1})
        if sample:
            embedding = sample.get("embedding")
            embedding_len = len(embedding) if isinstance(embedding, list) else None
            logging.info(
                "Sample doc: _id=%s path=%s embedding_len=%s",
                sample.get("_id"),
                (sample.get("meta") or {}).get("path"),
                embedding_len,
            )
    except Exception as e:
        logging.debug(f"Could not read collection diagnostics: {e}")

    try:
        indexes = list(_collection.list_search_indexes())
        if indexes:
            names = [idx.get("name") for idx in indexes if isinstance(idx, dict)]
            logging.info(f"Atlas Search indexes visible to this user: {names}")
        else:
            logging.info("No Atlas Search indexes returned by list_search_indexes().")
    except Exception as e:
        logging.debug(f"Could not list Atlas Search indexes: {e}")


def init_retriever(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    embed_model_name: str,
    vector_index_name: str = "vector_index",
    embedding_path: str = "embedding",
):
    """Initialize the MongoDB client and embedding model for retrieval."""
    global _client, _collection, _embedder, _embed_model_name, _vector_index_name, _embedding_path
    _client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=30_000,
        connectTimeoutMS=20_000,
        socketTimeoutMS=20_000,
    )
    try:
        _client.admin.command("ping")
    except Exception as e:
        hint = _mongo_connection_hint(e)
        if hint:
            logging.error(hint)
        raise RuntimeError(f"Could not connect to MongoDB ({_redact_mongo_uri(mongo_uri)}): {e}") from e
    _collection = _client[db_name][collection_name]
    _embed_model_name = embed_model_name
    _vector_index_name = vector_index_name or "vector_index"
    _embedding_path = embedding_path or "embedding"
    _try_log_collection_diagnostics()
    # Load embedding model
    logging.info(f"Loading embedding model '{embed_model_name}' for retrieval...")
    from sentence_transformers import SentenceTransformer
    _embedder = SentenceTransformer(embed_model_name)

def get_relevant_chunks(question: str, top_k: int = 5):
    """Retrieve top-k relevant code chunks for the given question."""
    if _client is None or _embedder is None:
        raise RuntimeError("Retriever not initialized. Call init_retriever() first.")
    # Embed the question
    query_vector = _embedder.encode([question])[0]
    query_vector = list(map(float, query_vector))

    num_candidates = max(100, top_k * 20)

    def _run(pipeline):
        try:
            return list(_collection.aggregate(pipeline)), None
        except Exception as e:
            return [], e

    # Prefer the dedicated $vectorSearch stage (current Atlas Vector Search).
    pipeline_vector_search = [
        {
            "$vectorSearch": {
                "index": _vector_index_name,
                "path": _embedding_path,
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": top_k, # mandatory in $vectorSearch
            }
        },
        {"$project": {"text": 1, "meta": 1, "score": {"$meta": "vectorSearchScore"}}},
        {"$limit": top_k},
    ]
    results, err = _run(pipeline_vector_search)
    if err:
        logging.debug(f"$vectorSearch failed: {err}")

    # Fallback to older Atlas Search syntax ($search + knnBeta).
    if not results:
        pipeline_knn_beta = [
            {
                "$search": {
                    "index": _vector_index_name,
                    "knnBeta": {
                        "path": _embedding_path,
                        "vector": query_vector,
                        "k": top_k,
                        "numCandidates": num_candidates,
                    },
                }
            },
            {"$project": {"text": 1, "meta": 1, "score": {"$meta": "searchScore"}}},
            {"$limit": top_k},
        ]
        results, err = _run(pipeline_knn_beta)
        if err:
            logging.debug(f"$search knnBeta failed: {err}")

    # Last-resort fallback: regex search in text so the app can still return *something*
    # while debugging Atlas Vector Search configuration/indexing issues.
    if not results:
        keywords = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question)]
        keywords = keywords[:8]
        if keywords:
            pattern = "|".join(re.escape(k) for k in keywords)
            pipeline_regex = [
                {"$match": {"text": {"$regex": pattern, "$options": "i"}}},
                {"$project": {"text": 1, "meta": 1}},
                {"$limit": top_k},
            ]
            results, err = _run(pipeline_regex)
            if err:
                logging.debug(f"Regex fallback failed: {err}")
    if not results:
        logging.info("No relevant chunks found for query.")
        logging.info(
            "Troubleshooting: verify an Atlas Vector Search index named '%s' exists and indexes '%s' as a %s-dim vector.",
            _vector_index_name,
            _embedding_path,
            len(query_vector),
        )
        return []
    logging.info(f"Retrieved {len(results)} relevant chunks for query.")
    # We will return the raw 'text' and 'meta' fields for each result
    chunks = []
    for doc in results:
        chunks.append({
            "text": doc.get("text", ""),
            "meta": doc.get("meta", {})
        })
    return chunks
