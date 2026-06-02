"""
Pydantic Schemas — DeepCheck-AI
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


# ---------------------------
# ENUMS
# ---------------------------
class ChunkStrategy(str, Enum):
    SENTENCE = "sentence"
    SLIDING_WINDOW = "sliding_window"


class AlgorithmType(str, Enum):
    SEMANTIC = "semantic"
    TRADITIONAL = "traditional"


# ---------------------------
# REQUEST MODELS
# ---------------------------
class AnalyzeRequest(BaseModel):
    """Legacy endpoint — retained for backward compatibility."""
    text1: str = Field(..., min_length=10)
    text2: str = Field(..., min_length=10)


class DocumentAnalyzeRequest(BaseModel):
    source_document: str = Field(..., min_length=50)
    suspicious_document: str = Field(..., min_length=50)
    chunk_strategy: ChunkStrategy = Field(default=ChunkStrategy.SENTENCE)
    window_size: Optional[int] = Field(default=30, ge=10, le=200)
    overlap: Optional[int] = Field(default=10, ge=0)
    algorithm: Optional[AlgorithmType] = Field(default=AlgorithmType.SEMANTIC)

    @model_validator(mode="after")
    def validate_window_params(self) -> "DocumentAnalyzeRequest":
        if self.chunk_strategy == ChunkStrategy.SLIDING_WINDOW:
            if self.overlap >= self.window_size:
                raise ValueError("overlap must be strictly less than window_size.")
        return self


# ---------------------------
# RESPONSE MODELS
# ---------------------------
class ChunkMatch(BaseModel):
    suspicious_chunk_index: int
    suspicious_chunk_text: str
    best_match_source_index: int
    best_match_source_text: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    verdict: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_filename: Optional[str] = Field(default="Direct Upload Input")
    source_document_id: Optional[str] = Field(default="N/A")


class DocumentAnalyzeResponse(BaseModel):
    chunk_strategy: ChunkStrategy
    total_suspicious_chunks: int
    overall_similarity: float = Field(..., ge=0.0, le=1.0)
    max_similarity: float = Field(..., ge=0.0, le=1.0)
    chunk_matches: List[ChunkMatch]
    knowledge_base_chunks_searched: Optional[int] = Field(default=0)
    source_filename: Optional[str] = Field(default="Original Source")
    suspicious_filename: Optional[str] = Field(default="Suspicious Submission")
    auto_ingested: Optional[bool] = Field(default=False)
    algorithm_used: Optional[str] = Field(default="semantic")


# ---------------------------
# CHROMADB SCHEMAS
# ---------------------------
class IngestResponse(BaseModel):
    filename: str
    document_id: str
    chunks_stored: int
    chunking_strategy: ChunkStrategy


class KnowledgeBaseStats(BaseModel):
    collection_name: str
    total_chunks: int
    persist_directory: str
    total_documents: Optional[int] = Field(default=0)
