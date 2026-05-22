# app/api/routes.py
import numpy as np
from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    AnalyzeRequest,
    DocumentAnalyzeRequest,
    DocumentAnalyzeResponse,
)

from app.services.chunking_service import get_chunks
from app.services.similarity_service import (
    compute_similarity,
    compute_chunk_similarity_matrix,
    compute_aggregate_scores,
)

router = APIRouter()


def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    return float(
        np.dot(v1, v2) /
        (np.linalg.norm(v1) * np.linalg.norm(v2))
    )


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/")
async def root():
    return {"message": "DeepCheck API is running"}


# ---------------------------------------------------------
# BASIC TEXT-TO-TEXT SIMILARITY ENDPOINT
# ---------------------------------------------------------

@router.post("/analyze")
async def analyze(payload: AnalyzeRequest):
    """
    Legacy endpoint — compares text blocks as whole strings.
    """
    try:
        # FIX: Uses the updated compute_similarity function under the hood
        similarity_score = compute_similarity(payload.text1, payload.text2)
        similarity_percentage = round(similarity_score * 100, 2)

        return {
            "similarity_score": similarity_percentage,
            "message": "Semantic similarity analysis completed."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# DOCUMENT-LEVEL CHUNK ANALYSIS ENDPOINT
# ---------------------------------------------------------

@router.post(
    "/document/analyze",
    response_model=DocumentAnalyzeResponse
)
async def analyze_documents(request: DocumentAnalyzeRequest):
    """
    Chunk-based plagiarism detection pipeline.
    """
    try:
        source_chunks = get_chunks(
            text=request.source_document,
            strategy=request.chunk_strategy,
            window_size=request.window_size,
            overlap=request.overlap,
        )

        suspicious_chunks = get_chunks(
            text=request.suspicious_document,
            strategy=request.chunk_strategy,
            window_size=request.window_size,
            overlap=request.overlap,
        )

        matches = compute_chunk_similarity_matrix(
            source_chunks,
            suspicious_chunks
        )

        aggregates = compute_aggregate_scores(matches)

        return DocumentAnalyzeResponse(
            chunk_strategy=request.chunk_strategy,
            total_suspicious_chunks=len(suspicious_chunks),
            overall_similarity=aggregates["overall_similarity"],
            max_similarity=aggregates["max_similarity"],
            chunk_matches=matches,
        )
        
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")