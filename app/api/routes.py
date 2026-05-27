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


# Basic Text-To-Text Similarity Endpoint

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


# Document-Level Chunk Analysis Endpoint

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


# File Upload Document Analysis Endpoint

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


# Unified Endpoint - accepts any mix of text / file inputs

async def _resolve_input(
    text: Optional[str],
    file: Optional[UploadFile],
    field_name: str,
) -> str:
    """
    Resolve a single input side (source or suspicious).

    Priority: if a file is provided, use the file (ignoring text).
    Otherwise fall back to text.  Raises if neither is supplied.
    """
    if file is not None:
        file_bytes = await _read_upload_file(file)
        return await parse_uploaded_file(
            file_bytes=file_bytes,
            filename=file.filename,
            content_type=file.content_type,
        )

    if text is not None and text.strip():
        return text

    raise HTTPException(
        status_code=400,
        detail=f"You must provide either '{field_name}_text' or '{field_name}_file'.",
    )


@router.post(
    "/document/analyze/unified",
    response_model=DocumentAnalyzeResponse,
)
async def analyze_unified(
    source_text: Optional[str] = Form(
        default=None,
        description="Plain text for the source/reference document.",
    ),
    suspicious_text: Optional[str] = Form(
        default=None,
        description="Plain text for the suspicious document.",
    ),
    source_file: Optional[UploadFile] = File(
        default=None,
        description="Uploaded source document (PDF, DOCX, or TXT).",
    ),
    suspicious_file: Optional[UploadFile] = File(
        default=None,
        description="Uploaded suspicious document (PDF, DOCX, or TXT).",
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
    Unified plagiarism-detection endpoint.

    Each side (source & suspicious) can be supplied as **either**
    a plain-text field **or** a file upload — mix and match freely.

    If both text and file are provided for the same side, the file
    takes priority.

    Delegates to the same chunked analysis pipeline used by
    /document/analyze and /document/analyze/upload.
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

        # ── Resolve each side to plain text ─────────────────────────────
        source_document = await _resolve_input(source_text, source_file, "source")
        suspicious_document = await _resolve_input(suspicious_text, suspicious_file, "suspicious")

        # ── Delegate to existing pipeline ───────────────────────────────
        request = DocumentAnalyzeRequest(
            source_document=source_document,
            suspicious_document=suspicious_document,
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