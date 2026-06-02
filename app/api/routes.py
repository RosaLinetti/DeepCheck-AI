# app/api/routes.py
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from typing import Optional

from app.api.schemas import (
    AnalyzeRequest,
    DocumentAnalyzeResponse,
    ChunkStrategy,
    ChunkMatch,
    IngestResponse,
    KnowledgeBaseStats,
)

from app.services.chunking_service import get_chunks
from app.services.similarity_service import compute_similarity
from app.services.document_parser import parse_uploaded_file, DocumentParseError

from app.services.chroma_service import (
    add_document_chunks,
    query_similar_chunks,
    get_collection_stats,
    get_all_documents,
    document_already_exists,
    generate_file_hash,
    delete_document,
)

from app.services.embedding_service import EmbeddingService
from app.services.classifier_service import classify_chunk

embedding_service = EmbeddingService()
router = APIRouter()


# ---------------------------
# UTILS
# ---------------------------
def _clamp_similarity_score(score: float) -> float:
    """Clamp similarity score to [0.0, 1.0] to prevent Pydantic validation errors."""
    return float(min(1.0, max(0.0, score)))


def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def lexical_similarity(text_a: str, text_b: str) -> float:
    """
    Traditional plagiarism: pure Jaccard overlap on word sets.
    Returns a float in [0.0, 1.0].
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)


async def _read_upload_file(file: UploadFile) -> bytes:
    try:
        return await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            await file.seek(0)
        except Exception:
            pass


# ---------------------------
# BASIC ROUTES
# ---------------------------
@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/")
def root():
    return {"message": "DeepCheck API running"}


# ---------------------------
# LEGACY TEXT SIMILARITY
# ---------------------------
@router.post("/analyze")
def analyze(payload: AnalyzeRequest):
    try:
        score = compute_similarity(payload.text1, payload.text2)
        return {
            "similarity_score": round(score * 100, 2),
            "message": "Analysis complete"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# 1V1 DOCUMENT ANALYSIS
# ---------------------------
@router.post("/document/analyze", response_model=DocumentAnalyzeResponse)
async def analyze_documents(
    source_file: UploadFile = File(None),
    suspicious_file: UploadFile = File(None),
    source_text: Optional[str] = Form(None),
    suspicious_text: Optional[str] = Form(None),
    chunk_strategy: str = Form(default="sentence"),
    window_size: int = Form(default=30),
    overlap: int = Form(default=10),
    algorithm: str = Form(default="semantic"),   # ── NEW: "semantic" | "traditional"
):
    if chunk_strategy == "sliding_window" and overlap >= window_size:
        raise HTTPException(status_code=400, detail="overlap must be < window_size")

    try:
        # ------ SOURCE INPUT ------
        source_filename = "Source Baseline"
        if source_file:
            source_bytes = await _read_upload_file(source_file)
            source_text_final = await parse_uploaded_file(
                file_bytes=source_bytes,
                filename=source_file.filename,
                content_type=source_file.content_type,
            )
            source_filename = source_file.filename
        elif source_text:
            source_text_final = source_text
        else:
            raise HTTPException(status_code=400, detail="Source input missing")

        # ------ SUSPICIOUS INPUT ------
        suspicious_filename = "Suspicious Submission"
        if suspicious_file:
            suspicious_bytes = await _read_upload_file(suspicious_file)
            suspicious_text_final = await parse_uploaded_file(
                file_bytes=suspicious_bytes,
                filename=suspicious_file.filename,
                content_type=suspicious_file.content_type,
            )
            suspicious_filename = suspicious_file.filename
        elif suspicious_text:
            suspicious_text_final = suspicious_text
        else:
            raise HTTPException(status_code=400, detail="Suspicious input missing")

        if not source_text_final.strip() or not suspicious_text_final.strip():
            raise HTTPException(status_code=422, detail="Empty document detected")

        # ------ CHUNKING ------
        source_chunks = get_chunks(source_text_final, chunk_strategy, window_size, overlap)
        suspicious_chunks = get_chunks(suspicious_text_final, chunk_strategy, window_size, overlap)

        if not source_chunks or not suspicious_chunks:
            raise HTTPException(status_code=422, detail="No chunks generated")

        # ------ MATCHING ------
        chunk_matches = []
        max_similarity = 0.0

        use_semantic = algorithm.lower() != "traditional"

        if use_semantic:
            # Full SBERT + ML hybrid path
            source_emb = embedding_service.get_embeddings_batch(source_chunks).tolist()
            suspicious_emb = embedding_service.get_embeddings_batch(suspicious_chunks).tolist()

            for i, (s_chunk, s_vec) in enumerate(zip(suspicious_chunks, suspicious_emb)):
                best_score = -1.0
                best_match = ""
                best_idx = 0

                for j, (src_chunk, src_vec) in enumerate(zip(source_chunks, source_emb)):
                    sim = cosine_similarity(s_vec, src_vec)
                    if sim > best_score:
                        best_score = sim
                        best_match = src_chunk
                        best_idx = j

                words_s = set(s_chunk.lower().split())
                words_t = set(best_match.lower().split())
                union = words_s | words_t
                lexical = len(words_s & words_t) / len(union) if union else 0.0

                length_ratio = (
                    min(len(words_s), len(words_t)) /
                    max(len(words_s), len(words_t))
                    if max(len(words_s), len(words_t)) > 0 else 0.0
                )

                verdict, confidence = classify_chunk(best_score, length_ratio, lexical)
                hybrid = _clamp_similarity_score(
                    (0.6 * best_score) + (0.2 * length_ratio) + (0.2 * lexical)
                )
                max_similarity = max(max_similarity, hybrid)

                chunk_matches.append(ChunkMatch(
                    suspicious_chunk_index=i,
                    suspicious_chunk_text=s_chunk,
                    best_match_source_index=best_idx,
                    best_match_source_text=best_match,
                    similarity_score=round(hybrid, 6),
                    verdict=verdict,
                    confidence=round(_clamp_similarity_score(confidence), 6),
                    source_filename=source_filename,
                    source_document_id="direct_1v1",
                ))

        else:
            # Traditional lexical-only path (no SBERT, fast)
            for i, s_chunk in enumerate(suspicious_chunks):
                best_score = 0.0
                best_match = ""
                best_idx = 0

                for j, src_chunk in enumerate(source_chunks):
                    score = lexical_similarity(s_chunk, src_chunk)
                    if score > best_score:
                        best_score = score
                        best_match = src_chunk
                        best_idx = j

                best_score = _clamp_similarity_score(best_score)
                max_similarity = max(max_similarity, best_score)

                if best_score >= 0.72:
                    verdict, confidence = "plagiarised", min(1.0, best_score + 0.05)
                elif best_score >= 0.45:
                    verdict, confidence = "suspicious", best_score
                else:
                    verdict, confidence = "original", 1.0 - best_score

                chunk_matches.append(ChunkMatch(
                    suspicious_chunk_index=i,
                    suspicious_chunk_text=s_chunk,
                    best_match_source_index=best_idx,
                    best_match_source_text=best_match,
                    similarity_score=round(best_score, 6),
                    verdict=verdict,
                    confidence=round(_clamp_similarity_score(confidence), 6),
                    source_filename=source_filename,
                    source_document_id="direct_1v1",
                ))

        overall = _clamp_similarity_score(
            sum(m.similarity_score for m in chunk_matches) / len(chunk_matches)
            if chunk_matches else 0.0
        )

        return DocumentAnalyzeResponse(
            chunk_strategy=ChunkStrategy(chunk_strategy),
            total_suspicious_chunks=len(suspicious_chunks),
            overall_similarity=round(overall, 6),
            max_similarity=round(max_similarity, 6),
            chunk_matches=chunk_matches,
            knowledge_base_chunks_searched=0,
            source_filename=source_filename,
            suspicious_filename=suspicious_filename,
            algorithm_used=algorithm.lower(),
        )

    except DocumentParseError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# ---------------------------
# CHROMA STATS
# ---------------------------
@router.get("/knowledge-base/stats", response_model=KnowledgeBaseStats)
def stats():
    try:
        return KnowledgeBaseStats(**get_collection_stats())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# INGEST DOCUMENT
# ---------------------------
@router.post("/document/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    chunk_strategy: str = Form("sentence"),
):
    try:
        file_bytes = await _read_upload_file(file)
        text = await parse_uploaded_file(file_bytes, file.filename, file.content_type)
        file_hash = generate_file_hash(text)

        if document_already_exists(file_hash):
            raise HTTPException(
                status_code=409,
                detail=f"{file.filename} already exists in ChromaDB",
            )

        chunks = get_chunks(text, chunk_strategy)
        embeddings = embedding_service.get_embeddings_batch(chunks).tolist()
        result = add_document_chunks(chunks, embeddings, file.filename, file_hash)

        return IngestResponse(
            filename=file.filename,
            document_id=result["document_id"],
            chunks_stored=result["chunks_added"],
            chunking_strategy=chunk_strategy,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# DELETE DOCUMENT  ── NEW ──
# ---------------------------
@router.delete("/document/delete")
def delete_doc(filename: str = Query(..., description="Filename to remove from ChromaDB")):
    """Remove all chunks belonging to a document from the knowledge base."""
    try:
        success = delete_document(filename)
        if not success:
            raise HTTPException(status_code=404, detail=f"'{filename}' not found in ChromaDB")
        return {"deleted": True, "filename": filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# DB SEARCH ANALYSIS
# ---------------------------
@router.post("/document/analyze/search", response_model=DocumentAnalyzeResponse)
async def analyze_db(
    file: UploadFile = File(...),
    chunk_strategy: str = Form("sentence"),
    top_k: int = Query(5, ge=1, le=20),
    algorithm: str = Form(default="semantic"),   # ── NEW: "semantic" | "traditional"
):
    try:
        db_stats = get_collection_stats()
        if db_stats["total_chunks"] == 0:
            raise HTTPException(status_code=400, detail="DB empty")

        file_bytes = await _read_upload_file(file)
        text = await parse_uploaded_file(file_bytes, file.filename, file.content_type)
        suspicious_chunks = get_chunks(text, chunk_strategy)

        use_semantic = algorithm.lower() != "traditional"

        chunk_matches = []
        max_sim = 0.0

        if use_semantic:
            # SBERT path: embed then query ChromaDB by vector
            suspicious_embeddings_np = embedding_service.get_embeddings_batch(suspicious_chunks)
            embeddings = suspicious_embeddings_np.tolist()
            results = query_similar_chunks(embeddings, top_k)

            for i, (chunk, matches) in enumerate(zip(suspicious_chunks, results)):
                for m in matches:
                    ref = m.get("chunk_text", "")
                    sim = m.get("similarity_score", 0.0)

                    words_s = set(chunk.lower().split())
                    words_r = set(ref.lower().split())
                    union = words_s | words_r
                    lexical = len(words_s & words_r) / len(union) if union else 0.0

                    length_ratio = (
                        min(len(words_s), len(words_r)) /
                        max(len(words_s), len(words_r))
                        if max(len(words_s), len(words_r)) > 0 else 0.0
                    )

                    verdict, confidence = classify_chunk(sim, length_ratio, lexical)
                    hybrid = _clamp_similarity_score(
                        (0.6 * sim) + (0.2 * length_ratio) + (0.2 * lexical)
                    )
                    max_sim = max(max_sim, hybrid)

                    chunk_matches.append(ChunkMatch(
                        suspicious_chunk_index=i,
                        suspicious_chunk_text=chunk,
                        best_match_source_index=0,
                        best_match_source_text=ref,
                        similarity_score=round(hybrid, 6),
                        verdict=verdict,
                        confidence=round(_clamp_similarity_score(confidence), 6),
                        source_filename=m.get("source_filename", "unknown"),
                        source_document_id=m.get("document_id", "db"),
                    ))

        else:
            # Traditional path: fetch all DB chunks and run Jaccard comparison
            from app.services.chroma_service import get_all_raw_chunks
            db_chunks = get_all_raw_chunks()   # list of {chunk_text, source_filename, document_id}

            if not db_chunks:
                raise HTTPException(status_code=400, detail="DB has no readable chunks")

            for i, s_chunk in enumerate(suspicious_chunks):
                best_score = 0.0
                best_ref = ""
                best_src = "unknown"
                best_doc_id = "db"

                for db_item in db_chunks:
                    score = lexical_similarity(s_chunk, db_item["chunk_text"])
                    if score > best_score:
                        best_score = score
                        best_ref = db_item["chunk_text"]
                        best_src = db_item.get("source_filename", "unknown")
                        best_doc_id = db_item.get("document_id", "db")

                best_score = _clamp_similarity_score(best_score)
                max_sim = max(max_sim, best_score)

                if best_score >= 0.72:
                    verdict, confidence = "plagiarised", min(1.0, best_score + 0.05)
                elif best_score >= 0.45:
                    verdict, confidence = "suspicious", best_score
                else:
                    verdict, confidence = "original", 1.0 - best_score

                chunk_matches.append(ChunkMatch(
                    suspicious_chunk_index=i,
                    suspicious_chunk_text=s_chunk,
                    best_match_source_index=0,
                    best_match_source_text=best_ref,
                    similarity_score=round(best_score, 6),
                    verdict=verdict,
                    confidence=round(_clamp_similarity_score(confidence), 6),
                    source_filename=best_src,
                    source_document_id=best_doc_id,
                ))

            embeddings = None  # not computed in traditional path

        overall = _clamp_similarity_score(
            sum(m.similarity_score for m in chunk_matches) / len(chunk_matches)
            if chunk_matches else 0.0
        )

        # ── AUTO-INGEST: if clean AND semantic path (embeddings available) ──
        AUTO_INGEST_THRESHOLD = 0.15
        auto_ingested = False

        if overall < AUTO_INGEST_THRESHOLD and use_semantic and embeddings is not None:
            try:
                file_hash = generate_file_hash(text)
                if not document_already_exists(file_hash):
                    add_document_chunks(
                        chunks=suspicious_chunks,
                        embeddings=embeddings,
                        source_filename=file.filename,
                        file_hash=file_hash,
                    )
                    auto_ingested = True
            except Exception as ingest_err:
                print(f"[INFO] Background auto-ingestion skipped: {ingest_err}")

        return DocumentAnalyzeResponse(
            chunk_strategy=ChunkStrategy(chunk_strategy),
            total_suspicious_chunks=len(suspicious_chunks),
            overall_similarity=round(overall, 6),
            max_similarity=round(max_sim, 6),
            chunk_matches=chunk_matches,
            knowledge_base_chunks_searched=int(db_stats["total_chunks"]),
            suspicious_filename=file.filename,
            auto_ingested=auto_ingested,
            algorithm_used=algorithm.lower(),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# LIST DOCUMENTS
# ---------------------------
@router.get("/knowledge-base/documents")
def list_documents():
    return {"documents": get_all_documents()}
