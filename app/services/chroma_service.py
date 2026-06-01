import chromadb
from chromadb.config import Settings
import uuid
import logging
from typing import Optional
import hashlib

logger = logging.getLogger(__name__)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_storage"
COLLECTION_NAME = "plagiarism_knowledge_base"
DEFAULT_TOP_K = 5


# ── CLIENT & COLLECTION INITIALIZATION ────────────────────────────────────────
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# Initialize once
_client = get_chroma_client()
_collection = get_or_create_collection(_client)

def generate_file_hash(text: str) -> str:
    """
    Creates a SHA256 hash from document content.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
#duplicate check
def document_already_exists(file_hash: str) -> bool:
    results = _collection.get(
        where={"file_hash": file_hash}
    )

    return len(results["ids"]) > 0

# ── LIST ALL DOCUMENTS ────────────────────────────────────────────────────────
def get_all_documents():
    """
    Returns every filename stored in ChromaDB.
    """

    try:
        results = _collection.get(include=["metadatas"])

        files = {}

        for meta in results.get("metadatas", []):
            filename = meta.get("source_filename", "unknown")

            if filename not in files:
                files[filename] = {
                    "chunks": 0,
                    "document_ids": set()
                }

            files[filename]["chunks"] += 1
            files[filename]["document_ids"].add(
                meta.get("document_id", "unknown")
            )

        output = []

        for filename, data in files.items():
            output.append({
                "filename": filename,
                "chunks": data["chunks"],
                "documents": len(data["document_ids"])
            })

        output.sort(key=lambda x: x["filename"].lower())

        return output

    except Exception as e:
        logger.error("Failed fetching documents: %s", str(e))
        return []


# ── DELETE DOCUMENT ───────────────────────────────────────────────────────────
def delete_document(filename: str) -> bool:
    try:
        results = _collection.get(
            where={"source_filename": filename}
        )

        ids = results.get("ids", [])

        if not ids:
            return False

        _collection.delete(ids=ids)

        logger.info(
            "Deleted %d chunks from '%s'",
            len(ids),
            filename
        )

        return True

    except Exception as e:
        logger.error("Delete failed: %s", str(e))
        return False


# ── DATA INGESTION ────────────────────────────────────────────────────────────
def add_document_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    source_filename: str,
    file_hash: str,
    document_id: Optional[str] = None,
) -> dict:
    if not chunks:
        raise ValueError("chunks list must not be empty.")

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
        )

    doc_id = document_id or str(uuid.uuid4())

    ids = [
        f"{doc_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source_filename": source_filename,
            "document_id": doc_id,
            "chunk_index": i,
            "file_hash": file_hash
        }
        for i in range(len(chunks))
    ]

    safe_embeddings = [
        [float(v) for v in vec]
        for vec in embeddings
    ]

    _collection.add(
        ids=ids,
        embeddings=safe_embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    logger.info(
        "Added %d chunks from '%s' (doc_id=%s)",
        len(chunks),
        source_filename,
        doc_id,
    )

    return {
        "document_id": doc_id,
        "chunks_added": len(chunks)
    }


# ── SEMANTIC SEARCH ───────────────────────────────────────────────────────────
def query_similar_chunks(
    query_embeddings: list[list[float]],
    top_k: int = DEFAULT_TOP_K,
) -> list[list[dict]]:

    if _collection.count() == 0:
        logger.warning("Collection is empty.")
        return [[] for _ in query_embeddings]

    safe_embeddings = [
        [float(v) for v in vec]
        for vec in query_embeddings
    ]

    results = _collection.query(
        query_embeddings=safe_embeddings,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    all_query_results = []

    for i in range(len(query_embeddings)):

        chunk_results = []

        docs = results["documents"][i]
        metas = results["metadatas"][i]
        distances = results["distances"][i]

        for doc, meta, dist in zip(
            docs,
            metas,
            distances
        ):
            similarity = float(1.0 - dist)

            chunk_results.append(
                {
                    "chunk_text": doc,
                    "source_filename": meta.get(
                        "source_filename",
                        "unknown"
                    ),
                    "document_id": meta.get(
                        "document_id",
                        "unknown"
                    ),
                    "chunk_index": int(
                        meta.get("chunk_index", -1)
                    ),
                    "similarity_score": round(
                        similarity,
                        6
                    ),
                }
            )

        all_query_results.append(chunk_results)

    return all_query_results


# ── COLLECTION STATS ──────────────────────────────────────────────────────────
def get_collection_stats() -> dict:
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": _collection.count(),
        "persist_directory": CHROMA_PERSIST_DIR,
    }