# app/api/routes.py
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

from app.api.schemas import (
    AnalyzeRequest,
    DocumentAnalyzeRequest,
    DocumentAnalyzeResponse,
    ChunkStrategy,
)

from app.services.chunking_service import get_chunks
from app.services.similarity_service import (
    compute_similarity,
    compute_chunk_similarity_matrix,
    compute_aggregate_scores,
)
from app.services.document_parser import (
    parse_uploaded_file,
    DocumentParseError,
    MAX_FILE_SIZE_BYTES,
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


# ---------------------------------------------------------
# FILE UPLOAD DOCUMENT ANALYSIS ENDPOINT
# ---------------------------------------------------------

async def _read_upload_file(upload: UploadFile) -> bytes:
    """
    Read an uploaded file in chunks to avoid loading huge files
    into memory all at once. Enforces MAX_FILE_SIZE_BYTES limit.
    """
    chunks = []
    total_size = 0
    chunk_size = 1024 * 1024  # 1 MB chunks

    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_FILE_SIZE_BYTES:
            raise DocumentParseError(
                f"File '{upload.filename}' exceeds the "
                f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB size limit."
            )
        chunks.append(chunk)

    return b"".join(chunks)


@router.post(
    "/document/analyze/upload",
    response_model=DocumentAnalyzeResponse,
)
async def analyze_uploaded_documents(
    source_file: UploadFile = File(
        ..., description="The original/reference document (PDF, DOCX, or TXT)."
    ),
    suspicious_file: UploadFile = File(
        ..., description="The document suspected of plagiarism (PDF, DOCX, or TXT)."
    ),
    chunk_strategy: Optional[str] = Form(
        default="sentence",
        description="Chunking strategy: 'sentence' or 'sliding_window'.",
    ),
    window_size: Optional[int] = Form(
        default=30,
        description="Token window size (only for sliding_window strategy).",
    ),
    overlap: Optional[int] = Form(
        default=10,
        description="Token overlap between windows (only for sliding_window).",
    ),
):
    """
    Upload-based plagiarism detection pipeline.

    Accepts two PDF, DOCX, or TXT files, extracts text from each,
    then delegates to the existing analyze_documents endpoint.

    Supports large documents (up to 50 MB per file).
    """
    try:
        # ── Validate chunk_strategy ─────────────────────────────────────
        try:
            strategy = ChunkStrategy(chunk_strategy)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chunk_strategy '{chunk_strategy}'. "
                       f"Must be 'sentence' or 'sliding_window'.",
            )

        # ── Read uploaded files ─────────────────────────────────────────
        source_bytes = await _read_upload_file(source_file)
        suspicious_bytes = await _read_upload_file(suspicious_file)

        # ── Extract text ────────────────────────────────────────────────
        source_text = await parse_uploaded_file(
            file_bytes=source_bytes,
            filename=source_file.filename,
            content_type=source_file.content_type,
        )
        suspicious_text = await parse_uploaded_file(
            file_bytes=suspicious_bytes,
            filename=suspicious_file.filename,
            content_type=suspicious_file.content_type,
        )

        # ── Delegate to the existing analyze_documents pipeline ─────────
        request = DocumentAnalyzeRequest(
            source_document=source_text,
            suspicious_document=suspicious_text,
            chunk_strategy=strategy,
            window_size=window_size,
            overlap=overlap,
        )

        return await analyze_documents(request)

    except DocumentParseError as parse_err:
        raise HTTPException(status_code=400, detail=str(parse_err))
    except HTTPException:
        raise
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")