import chromadb
from chromadb.config import Settings
import uuid
import logging
import hashlib
from typing import Optional, List

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = "./chroma_storage"
COLLECTION_NAME = "plagiarism_knowledge_base"
DEFAULT_TOP_K = 5

# ---------------------------
# CLIENT SETUP
# ---------------------------

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Create a persistent ChromaDB client. 
    Telemetry is disabled to avoid sending usage statistics.
    """
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.PersistentClient):
    """ Retrieve the plagiarism collection or create it if it does not exist."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

# Shared collection instance used by repository operations.
_client = get_chroma_client()
_collection = get_or_create_collection(_client)


# ---------------------------
# HASHING
# ---------------------------

"""used for duplicate detection during ingestion"""

def generate_file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() 


# ---------------------------
# DUPLICATE CHECK
# ---------------------------

def document_already_exists(file_hash: str) -> bool:
    try:
        results = _collection.get(
            where={"file_hash": file_hash},
            include=["metadatas"],
        )
        return bool(results.get("ids"))
    except Exception as e:
        logger.error(f"Duplicate check failed: {e}")
        return False


# ---------------------------
# LIST DOCUMENTS  ( with summary)
# ---------------------------

def get_all_documents() -> list:
    """Return one summary row per unique source filename."""
    try:
        results = _collection.get(include=["metadatas"])
        files: dict = {}
        for meta in results.get("metadatas", []):
            if not meta:
                continue
            filename = meta.get("source_filename", "unknown")
            doc_id   = meta.get("document_id", "unknown")
            if filename not in files:
                files[filename] = {"chunks": 0, "document_ids": set()}
            files[filename]["chunks"] += 1
            files[filename]["document_ids"].add(doc_id)

        output = [
            {
                "filename":  fname,
                "chunks":    data["chunks"],
                "documents": len(data["document_ids"]),
            }
            for fname, data in files.items()
        ]
        output.sort(key=lambda x: x["filename"].lower())
        return output
    except Exception as e:
        logger.error(f"Failed fetching documents: {e}")
        return []


# ---------------------------
# GET ALL RAW CHUNKS 
# ---------------------------

def get_all_raw_chunks() -> List[dict]:
    """Return all stored chunks for lexical/text-based search."""
    try:
        results = _collection.get(include=["documents", "metadatas"])
        chunks = []
        for doc, meta in zip(
            results.get("documents", []),
            results.get("metadatas", []),
        ):
            if doc:
                chunks.append({
                    "chunk_text":      doc,
                    "source_filename": (meta or {}).get("source_filename", "unknown"),
                    "document_id":     (meta or {}).get("document_id", "unknown"),
                    "chunk_index":     int((meta or {}).get("chunk_index", -1)),
                })
        return chunks
    except Exception as e:
        logger.error(f"get_all_raw_chunks failed: {e}")
        return []


# ---------------------------
# DELETE DOCUMENT
# ---------------------------

def delete_document(filename: str) -> bool:
    try:
        results = _collection.get(where={"source_filename": filename})
        ids = results.get("ids", [])
        if not ids:
            return False
        _collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} chunks for '{filename}'")
        return True
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return False


# ---------------------------
# INGEST
# ---------------------------

def add_document_chunks(
    chunks: List[str],
    embeddings: List[List[float]],
    source_filename: str,
    file_hash: str,
    document_id: Optional[str] = None,
) -> dict:
    """ Store document chunks in the vector database."""

    if not chunks:
        raise ValueError("Chunks cannot be empty")
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings size mismatch")

    doc_id    = document_id or str(uuid.uuid4())
    ids       = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source_filename": source_filename,
            "document_id":     doc_id,
            "chunk_index":     i,
            "file_hash":       file_hash,
        }
        for i in range(len(chunks))
    ]
    safe_embeddings = [[float(v) for v in vec] for vec in embeddings]

    _collection.add(
        ids=ids,
        embeddings=safe_embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    logger.info(f"Added {len(chunks)} chunks from '{source_filename}' (doc_id={doc_id})")
    return {"document_id": doc_id, "chunks_added": len(chunks)}


# ---------------------------
# SEMANTIC SEARCH
# ---------------------------

def query_similar_chunks(
    query_embeddings: List[List[float]],
    top_k: int = DEFAULT_TOP_K,
) -> List[List[dict]]:
    if _collection.count() == 0:
        return [[] for _ in query_embeddings]

    safe_embeddings = [[float(v) for v in vec] for vec in query_embeddings]
    results = _collection.query(
        query_embeddings=safe_embeddings,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    all_results = []
    for i in range(len(query_embeddings)):
        chunk_results = []
        for doc, meta, dist in zip(
            results["documents"][i],
            results["metadatas"][i],
            results["distances"][i],
        ):
            similarity = _clamp(1.0 - float(dist))
            chunk_results.append({
                "chunk_text":      doc,
                "source_filename": meta.get("source_filename", "unknown"),
                "document_id":     meta.get("document_id", "unknown"),
                "chunk_index":     int(meta.get("chunk_index", -1)),
                "similarity_score": round(similarity, 6),
            })
        all_results.append(chunk_results)
    return all_results


def _clamp(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def get_collection_stats() -> dict:
    """
    Return chunk and document counts for the collection.
    """
    total_chunks = _collection.count()
    total_documents = 0

    if total_chunks > 0:
        try:
            results = _collection.get(include=["metadatas"])
            doc_ids = {
                meta.get("document_id")
                for meta in results.get("metadatas", [])
                if meta and meta.get("document_id")
            }
            total_documents = len(doc_ids)
        except Exception as e:
            logger.error(f"Could not count unique documents: {e}")

    return {
        "collection_name":  COLLECTION_NAME,
        "total_chunks":     total_chunks,
        "persist_directory": CHROMA_PERSIST_DIR,
        "total_documents":  total_documents,
    }
