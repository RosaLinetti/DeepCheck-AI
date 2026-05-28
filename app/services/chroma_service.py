# app/services/chroma_service.py
import chromadb
from chromadb.config import Settings
import uuid
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_storage"
COLLECTION_NAME = "plagiarism_knowledge_base"
DEFAULT_TOP_K = 5


# ── CLIENT & COLLECTION INITIALIZATION ────────────────────────────────────────
def get_chroma_client() -> chromadb.PersistentClient:
    """Creates a local directory on disk so database items survive application restarts."""
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Fetches or builds the target vector library using HNSW Cosine metrics."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# Module-level initialization singletons
_client: chromadb.PersistentClient = get_chroma_client()
_collection: chromadb.Collection = get_or_create_collection(_client)


# ── DATA INGESTION OPERATION ──────────────────────────────────────────────────
def add_document_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    source_filename: str,
    document_id: Optional[str] = None,
) -> dict:
    """Stores reference text fragments, vector arrays, and origin source filenames."""
    if not chunks:
        raise ValueError("chunks list must not be empty.")
    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings.")

    doc_id = document_id or str(uuid.uuid4())
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

    metadatas = [
        {
            "source_filename": source_filename,
            "document_id": doc_id,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    # Convert numeric types to basic Python floats to guarantee stability inside Chroma
    safe_embeddings = [[float(v) for v in vec] for vec in embeddings]

    _collection.add(
        ids=ids,
        embeddings=safe_embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    logger.info("Added %d chunks from '%s' (doc_id=%s).", len(chunks), source_filename, doc_id)
    return {"document_id": doc_id, "chunks_added": len(chunks)}


# ── SEMANTIC RETRIEVAL SEARCH OPERATION ───────────────────────────────────────
def query_similar_chunks(
    query_embeddings: list[list[float]],
    top_k: int = DEFAULT_TOP_K,
) -> list[list[dict]]:
    """
    For each suspicious vector chunk, pulls down the matching data points.
    Transforms raw metrics into a clean list of dictionaries mapped to our schemas.
    """
    if _collection.count() == 0:
        logger.warning("ChromaDB library collection is completely empty.")
        return [[] for _ in query_embeddings]

    safe_embeddings = [[float(v) for v in vec] for vec in query_embeddings]

    results = _collection.query(
        query_embeddings=safe_embeddings,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    all_query_results: list[list[dict]] = []

    for i in range(len(query_embeddings)):
        chunk_results: list[dict] = []

        docs = results["documents"][i]
        metas = results["metadatas"][i]
        distances = results["distances"][i]

        for doc, meta, dist in zip(docs, metas, distances):
            # Transform Cosine Distance to Cosine Similarity: S = 1.0 - D
            similarity = float(1.0 - dist)

            chunk_results.append(
                {
                    "chunk_text": doc,
                    "source_filename": meta.get("source_filename", "unknown"),
                    "document_id": meta.get("document_id", "unknown"),
                    "chunk_index": int(meta.get("chunk_index", -1)),
                    "similarity_score": round(similarity, 6),
                }
            )
        all_query_results.append(chunk_results)

    return all_query_results


# ── UTILITY RECONSTRUCTORS ────────────────────────────────────────────────────
def get_collection_stats() -> dict:
    """Tracks current repository volume levels."""
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": _collection.count(),
        "persist_directory": CHROMA_PERSIST_DIR,
    }