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
)

from app.services.embedding_service import EmbeddingService
from app.services.classifier_service import classify_chunk

embedding_service = EmbeddingService()
router = APIRouter()

#utils
def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0

    return float(np.dot(v1, v2) / denom)


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


#basic routes
@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/")
def root():
    return {"message": "DeepCheck API running"}


#lexacy text similarity
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



# 1 v1 document analysis (file+text hybrid)
@router.post("/document/analyze", response_model=DocumentAnalyzeResponse)
async def analyze_documents(
    source_file: UploadFile = File(None),
    suspicious_file: UploadFile = File(None),
    source_text: Optional[str] = Form(None),
    suspicious_text: Optional[str] = Form(None),
    chunk_strategy: str = Form(default="sentence"),
    window_size: int = Form(default=30),
    overlap: int = Form(default=10),
):

    if chunk_strategy == "sliding_window" and overlap >= window_size:
        raise HTTPException(status_code=400, detail="overlap must be < window_size")

    try:
        #source input
        if source_file:
            source_bytes = await _read_upload_file(source_file)
            source_text_final = await parse_uploaded_file(
                file_bytes=source_bytes,
                filename=source_file.filename,
                content_type=source_file.content_type,
            )
        elif source_text:
            source_text_final = source_text
        else:
            raise HTTPException(status_code=400, detail="Source input missing")

        #suspicious input
        if suspicious_file:
            suspicious_bytes = await _read_upload_file(suspicious_file)
            suspicious_text_final = await parse_uploaded_file(
                file_bytes=suspicious_bytes,
                filename=suspicious_file.filename,
                content_type=suspicious_file.content_type,
            )
        elif suspicious_text:
            suspicious_text_final = suspicious_text
        else:
            raise HTTPException(status_code=400, detail="Suspicious input missing")

        if not source_text_final.strip() or not suspicious_text_final.strip():
            raise HTTPException(status_code=422, detail="Empty document detected")

        #chunking
        source_chunks = get_chunks(source_text_final, chunk_strategy, window_size, overlap)
        suspicious_chunks = get_chunks(suspicious_text_final, chunk_strategy, window_size, overlap)

        if not source_chunks or not suspicious_chunks:
            raise HTTPException(status_code=422, detail="No chunks generated")

        #embeddings
        source_emb = embedding_service.get_embeddings_batch(source_chunks).tolist()
        suspicious_emb = embedding_service.get_embeddings_batch(suspicious_chunks).tolist()

        #matching
        chunk_matches = []
        max_similarity = 0.0

        for i, (s_chunk, s_vec) in enumerate(zip(suspicious_chunks, suspicious_emb)):
            best_score = -1
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
                max(len(words_s), len(words_t)) if max(len(words_s), len(words_t)) > 0 else 0.0
            )

            verdict, confidence = classify_chunk(best_score, length_ratio, lexical)

            hybrid = (0.6 * best_score) + (0.2 * length_ratio) + (0.2 * lexical)
            max_similarity = max(max_similarity, hybrid)

            chunk_matches.append(
                ChunkMatch(
                    suspicious_chunk_index=i,
                    suspicious_chunk_text=s_chunk,
                    best_match_source_index=best_idx,
                    best_match_source_text=best_match,
                    similarity_score=round(hybrid, 6),
                    verdict=verdict,
                    confidence=round(confidence, 6),
                    source_filename="hybrid_input",
                    source_document_id="direct_1v1"
                )
            )

        overall = (
            sum(m.similarity_score for m in chunk_matches) / len(chunk_matches)
            if chunk_matches else 0.0
        )

        return DocumentAnalyzeResponse(
            chunk_strategy=ChunkStrategy(chunk_strategy),
            total_suspicious_chunks=len(suspicious_chunks),
            overall_similarity=round(overall, 6),
            max_similarity=round(max_similarity, 6),
            chunk_matches=chunk_matches,
            knowledge_base_chunks_searched=0
        )

    except DocumentParseError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")




@router.get("/knowledge-base/stats", response_model=KnowledgeBaseStats)
async def knowledge_base_stats():
   """Returns the volume metrics and local directory status of ChromaDB."""
   try:
       stats = get_collection_stats()
       return KnowledgeBaseStats(**stats)
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Database status error: {str(e)}")




@router.post("/document/ingest", response_model=IngestResponse)
async def ingest_reference_document(
   file: UploadFile = File(..., description="Target reference material (PDF, DOCX, TXT)"),
   chunk_strategy: Optional[str] = Form(default="sentence"),
   window_size: Optional[int] = Form(default=30),
   overlap: Optional[int] = Form(default=10),
):
   """Parses a reference document, chunks it, embeds vectors, and stores in ChromaDB."""
   if chunk_strategy == "sliding_window" and overlap >= window_size:
       raise HTTPException(
           status_code=400,
           detail="Configuration Error: overlap must be strictly less than window_size."
       )


   try:
       file_bytes = await _read_upload_file(file)
       raw_text = await parse_uploaded_file(
           file_bytes=file_bytes,
           filename=file.filename,
           content_type=file.content_type,
       )
       if not raw_text.strip():
           raise HTTPException(status_code=422, detail="Extraction yielded empty text blocks.")


       chunks = get_chunks(text=raw_text, strategy=chunk_strategy, window_size=window_size, overlap=overlap)
       if not chunks:
           raise HTTPException(status_code=422, detail="Subdivision routine generated zero blocks.")


       embeddings_np = embedding_service.get_embeddings_batch(chunks)
       embeddings = embeddings_np.tolist()


       result = add_document_chunks(chunks=chunks, embeddings=embeddings, source_filename=file.filename)


       return IngestResponse(
           filename=file.filename,
           document_id=result["document_id"],
           chunks_stored=result["chunks_added"],
           chunking_strategy=chunk_strategy,
       )
   except DocumentParseError as parse_err:
       raise HTTPException(status_code=400, detail=str(parse_err))
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Ingestion Pipeline Fault: {str(e)}")




@router.post("/document/analyze/search", response_model=DocumentAnalyzeResponse)
async def analyze_suspicious_document_against_db(
   file: UploadFile = File(..., description="Student submission document targeted for scanning"),
   chunk_strategy: Optional[str] = Form(default="sentence"),
   window_size: Optional[int] = Form(default=30),
   overlap: Optional[int] = Form(default=10),
   top_k: int = Query(default=5, ge=1, le=20),
):
   """Analyze suspicious submission against the entire ChromaDB knowledge base."""
   if chunk_strategy == "sliding_window" and overlap >= window_size:
       raise HTTPException(
           status_code=400,
           detail="Configuration Error: overlap must be strictly less than window_size."
       )


   stats = get_collection_stats()
   if stats["total_chunks"] == 0:
       raise HTTPException(
           status_code=400,
           detail="Knowledge base is empty. Run /document/ingest first."
       )


   try:
       file_bytes = await _read_upload_file(file)
       raw_text = await parse_uploaded_file(
           file_bytes=file_bytes,
           filename=file.filename,
           content_type=file.content_type,
       )
       if not raw_text.strip():
           raise HTTPException(status_code=422, detail="Target text extraction returned empty payload.")


       suspicious_chunks = get_chunks(text=raw_text, strategy=chunk_strategy, window_size=window_size, overlap=overlap)
       if not suspicious_chunks:
           raise HTTPException(status_code=422, detail="Chunk processor generated empty list.")


       suspicious_embeddings_np = embedding_service.get_embeddings_batch(suspicious_chunks)
       suspicious_embeddings = suspicious_embeddings_np.tolist()


       retrieved = query_similar_chunks(query_embeddings=suspicious_embeddings, top_k=top_k)


       chunk_matches = []
       max_similarity_detected = 0.0


       for susp_idx, (susp_chunk, top_matches) in enumerate(zip(suspicious_chunks, retrieved)):
           for match in top_matches:
               # Defensive check: Extract safely whether match is an object or a dictionary
               if hasattr(match, "get"): # It's a dictionary
                   ref_chunk = match.get("chunk_text", "")
                   chroma_sim = match.get("similarity_score", 0.0)
                   src_chunk_idx = match.get("chunk_index", 0)
                   src_filename = match.get("source_filename", "unknown")
                   src_doc_id = match.get("document_id", "unknown")
               else: # It's an object/Pydantic model
                   ref_chunk = getattr(match, "chunk_text", "")
                   chroma_sim = getattr(match, "similarity_score", 0.0)
                   src_chunk_idx = getattr(match, "chunk_index", 0)
                   src_filename = getattr(match, "source_filename", "unknown")
                   src_doc_id = getattr(match, "document_id", "unknown")


               # Feature generation
               words_susp = set(susp_chunk.lower().split())
               words_src = set(ref_chunk.lower().split())
               union = words_susp | words_src
               lexical_overlap = len(words_susp & words_src) / len(union) if union else 0.0
               length_ratio = min(len(words_susp), len(words_src)) / max(len(words_susp), len(words_src)) if union else 0.0


               # ML Classifier block evaluation
               verdict, confidence = classify_chunk(
                   cosine_score=chroma_sim,
                   length_ratio=length_ratio,
                   lexical_overlap=lexical_overlap,
               )


               # Weighting score generation
               hybrid_score = (0.6 * chroma_sim) + (0.2 * length_ratio) + (0.2 * lexical_overlap)
              
               # Force the final score to cap at exactly 1.0 to prevent floating point overflows
               hybrid_score = min(float(hybrid_score), 1.0)
              
               if hybrid_score > max_similarity_detected:
                   max_similarity_detected = hybrid_score


               chunk_matches.append(
                   ChunkMatch(
                       suspicious_chunk_index=int(susp_idx),
                       suspicious_chunk_text=susp_chunk,
                       best_match_source_index=int(src_chunk_idx),
                       best_match_source_text=ref_chunk,
                       similarity_score=round(float(hybrid_score), 6),
                       verdict=verdict,
                       confidence=round(float(confidence), 6),
                       source_filename=src_filename,
                       source_document_id=str(src_doc_id)
                   )
               )


       # Calculate metrics safely
       overall_score = 0.0
       if chunk_matches:
           overall_score = round(sum(m.similarity_score for m in chunk_matches) / len(chunk_matches), 6)


       # Safe parsing fallback for ChunkStrategy Enum
       try:
           resolved_strategy = ChunkStrategy(chunk_strategy)
       except ValueError:
           # Fallback to whatever your default schema choice is if matching fails
           resolved_strategy = list(ChunkStrategy)[0]


       return DocumentAnalyzeResponse(
           chunk_strategy=resolved_strategy,
           total_suspicious_chunks=int(len(suspicious_chunks)),
           overall_similarity=overall_score,
           max_similarity=round(max_similarity_detected, 6),
           chunk_matches=chunk_matches,
           knowledge_base_chunks_searched=int(stats["total_chunks"])
       )


   except DocumentParseError as parse_err:
       raise HTTPException(status_code=400, detail=str(parse_err))
   except Exception as e:
       # Crucial tip: Temporarily print the error trace out to your terminal logs
       # so you can see exactly what line broke if another hidden fault exists!
       import traceback
       traceback.print_exc()
       raise HTTPException(status_code=500, detail=f"Analysis Engine Error: {str(e)}")
