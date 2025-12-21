import os
import ast
import hashlib
import re
from datetime import datetime
from typing import Optional
from . import utils
from .utils import count_tokens, split_chunks_by_tokens
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import logging

def _redact_mongo_uri(uri: str) -> str:
    return re.sub(r"(?<=://)([^:@/]+):([^@/]+)@", r"\1:***@", uri)

def _mongo_connection_hint(error: Exception) -> Optional[str]:
    message = str(error)
    if "SSL handshake failed" in message or "TLSV1_ALERT_INTERNAL_ERROR" in message:
        return (
            "MongoDB TLS handshake failed (not related to an empty DB/collection). "
            "Re-copy the official Atlas connection string (Connect → Drivers) and ensure your current IP "
            "is allowed in Atlas (Network Access). Avoid custom DNS/CNAME aliases for the cluster hostname."
        )
    return None

def index_codebase(repo_path: str, db_uri: str, db_name: str, collection_name: str, embed_model_name: str, chunk_token_target: int, chunk_overlap_tokens: int):
    """Walk through the Ultralytics repository and index specified directories into MongoDB."""
    # Connect to MongoDB
    client = MongoClient(
        db_uri,
        server_api=ServerApi("1"),
        serverSelectionTimeoutMS=30_000,
        connectTimeoutMS=20_000,
        socketTimeoutMS=20_000,
    )
    try:
        client.admin.command("ping")
        logging.info("Connected to MongoDB (ping OK).")
    except Exception as e:
        logging.error(f"Could not connect to MongoDB ({_redact_mongo_uri(db_uri)}): {e}")
        hint = _mongo_connection_hint(e)
        if hint:
            logging.error(hint)
        client.close()

    db = client[db_name]
    coll = db[collection_name]
    # Clear existing documents in collection (to avoid duplicates)
    try:
        coll.delete_many({})  # consider indexing by path if partial update needed
    except Exception as e:
        logging.error(f"Could not clear existing collection data: {e}")
        
    
    # Load embedding model
    logging.info(f"Loading embedding model '{embed_model_name}' for indexing...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(embed_model_name)
    
    # Define which subdirectories to index
    subdirs = ["ultralytics/models", "ultralytics/engine", "ultralytics/data"]
    total_files = 0
    total_chunks = 0
    for subdir in subdirs:
        dir_path = os.path.join(repo_path, subdir)
        if not os.path.isdir(dir_path):
            continue
        for root, _, files in os.walk(dir_path):
            for filename in files:
                if not filename.endswith(".py"):
                    continue
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, repo_path)
                # Read file content
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source = f.read()
                except Exception as e:
                    logging.error(f"Could not read file {file_path}: {e}")
                    continue
                total_files += 1
                # Parse file into AST to identify top-level chunks
                try:
                    tree = ast.parse(source)
                except Exception as e:
                    logging.error(f"AST parse failed for {file_path}: {e}")
                    # If parse fails (e.g., syntax error), index entire file as one chunk
                    chunk_list = [(1, len(source.splitlines()), None, source)]
                else:
                    chunk_list = []
                    prev_end_line = 0
                    for node in tree.body:
                        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                            start = node.lineno
                            end = node.end_lineno
                            # If there's a gap between previous node and this node, that's global code chunk
                            if start - 1 > prev_end_line:
                                gap_start = prev_end_line + 1
                                gap_end = start - 1
                                gap_text = utils.get_text_segment(source, gap_start, gap_end)
                                if gap_text.strip():
                                    chunk_list.append((gap_start, gap_end, None, gap_text))
                            # Determine symbol name (if any)
                            symbol = None
                            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                                symbol = f"function {node.name}"
                            elif isinstance(node, ast.ClassDef):
                                symbol = f"class {node.name}"
                            else:
                                # For other top-level (e.g., import, assign), treat them as part of module code
                                symbol = None
                            text_segment = utils.get_text_segment(source, start, end)
                            chunk_list.append((start, end, symbol, text_segment))
                            prev_end_line = end
                    # Any leftover lines after the last node
                    last_line = len(source.splitlines())
                    if prev_end_line < last_line:
                        gap_start = prev_end_line + 1
                        gap_end = last_line
                        gap_text = utils.get_text_segment(source, gap_start, gap_end)
                        if gap_text.strip():
                            chunk_list.append((gap_start, gap_end, None, gap_text))
                # Further split chunks that are too large
                refined_chunks = []
                for (cstart, cend, csymbol, ctext) in chunk_list:
                    # Determine token length
                    tok_count = count_tokens(ctext)
                    if tok_count > chunk_token_target:
                        # Split into smaller chunks with overlap
                        sub_chunks = split_chunks_by_tokens(ctext, cstart, chunk_token_target, chunk_overlap_tokens)
                        for sub_text, sub_start, sub_end in sub_chunks:
                            refined_chunks.append((sub_start, sub_end, csymbol, sub_text))
                    else:
                        refined_chunks.append((cstart, cend, csymbol, ctext))
                # Embed and upsert each chunk
                if not refined_chunks:
                    continue
                texts = [c[3] for c in refined_chunks]
                embeddings = embedder.encode(texts, batch_size=16, show_progress_bar=False)
                # Ensure embeddings are Python lists (not numpy arrays)
                embeddings = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
                docs = []
                norm_path = rel_path.replace('\\', '/')
                for idx, (cstart, cend, csymbol, ctext) in enumerate(refined_chunks):
                    meta = {
                        "path": norm_path,
                        "symbol": csymbol,
                        "start_line": cstart,
                        "end_line": cend,
                        "language": "python",
                        "chunk_id": idx,
                        "updated_at": datetime.utcnow()
                    }
                    doc = {
                        "_id": f"{norm_path}:{cstart}-{cend}",
                        "text": ctext,
                        "embedding": embeddings[idx],
                        "meta": meta
                    }
                    docs.append(doc)
                # Insert in bulk
                try:
                    coll.insert_many(docs, ordered=False)
                except Exception as e:
                    logging.error(f"Error inserting documents for file {rel_path}: {e}")
                total_chunks += len(docs)
    logging.info(f"Indexed {total_files} files with {total_chunks} chunks.")
    client.close()
