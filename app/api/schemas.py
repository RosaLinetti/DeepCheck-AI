"""
Pydantic Schemas — DeepCheck-AI
Validates incoming API request bodies and structures outgoing responses.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class ChunkStrategy(str, Enum):
    SENTENCE = "sentence"
    SLIDING_WINDOW = "sliding_window"


# ── Request Models ─────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """
    Legacy endpoint schema — retained for backward compatibility.
    Used by the basic /analyze route (whole-text vs. whole-text).
    """
    text1: str = Field(..., min_length=10, description="First text for comparison.")
    text2: str = Field(..., min_length=10, description="Second text for comparison.")


class DocumentAnalyzeRequest(BaseModel):
    """
    Primary schema for the chunked plagiarism pipeline.

    Accepts two full documents and the desired chunking strategy.
    Sliding window parameters are optional — only applied when
    chunk_strategy == SLIDING_WINDOW.
    """
    source_document: str = Field(
        ...,
        min_length=50,
        description="The original/reference document to check against."
    )
    suspicious_document: str = Field(
        ...,
        min_length=50,
        description="The document suspected of plagiarism."
    )
    chunk_strategy: ChunkStrategy = Field(
        default=ChunkStrategy.SENTENCE,
        description="Chunking strategy: 'sentence' or 'sliding_window'."
    )
    window_size: Optional[int] = Field(
        default=30,
        ge=10,
        le=200,
        description="Token window size (only for sliding_window strategy)."
    )
    overlap: Optional[int] = Field(
        default=10,
        ge=0,
        description="Token overlap between consecutive windows (only for sliding_window)."
    )

    @model_validator(mode="after")
    def validate_window_params(self) -> "DocumentAnalyzeRequest":
        if self.chunk_strategy == ChunkStrategy.SLIDING_WINDOW:
            if self.overlap >= self.window_size:
                raise ValueError(
                    "overlap must be strictly less than window_size."
                )
        return self


# ── Response Models ────────────────────────────────────────────────────────────

class ChunkMatch(BaseModel):
    """Represents a single suspicious chunk matched against the source."""
    suspicious_chunk_index: int
    suspicious_chunk_text: str
    best_match_source_index: int
    best_match_source_text: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    verdict: str  # "plagiarised", "suspicious", or "original"
    confidence: float = Field(..., ge=0.0, le=1.0)

class DocumentAnalyzeResponse(BaseModel):
    """
    Full pipeline response returned by /document/analyze.

    Fields:
        chunk_strategy:         Strategy used for this analysis run.
        total_suspicious_chunks: Total number of chunks extracted from suspicious doc.
        overall_similarity:     Mean similarity across all suspicious chunks.
        max_similarity:         Highest single-chunk similarity score detected.
        chunk_matches:          Per-chunk comparison details (index, text, score).
    """
    chunk_strategy: ChunkStrategy
    total_suspicious_chunks: int
    overall_similarity: float
    max_similarity: float
    chunk_matches: List[ChunkMatch]