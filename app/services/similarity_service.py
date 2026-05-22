# app/services/similarity_service.py
"""
Similarity Service — DeepCheck-AI
Handles:
  - Legacy: whole-text cosine similarity (backward compat)
  - Primary: chunk-matrix cosine similarity with index tracking
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

from app.services.embedding_service import EmbeddingService
from app.api.schemas import ChunkMatch


# Module-level singleton — avoids reloading the SBERT model on every call
_embedding_service = EmbeddingService()


def compute_similarity(text1: str, text2: str) -> float:
    """
    Legacy whole-text cosine similarity.
    Retained for the basic /analyze route.

    Returns:
        Float in [0.0, 1.0] representing semantic similarity.
    """
    vec1 = _embedding_service.get_embedding(text1)
    vec2 = _embedding_service.get_embedding(text2)
    score = cosine_similarity([vec1], [vec2])[0][0]
    return float(round(score, 4))


def compute_chunk_similarity_matrix(
    source_chunks: List[str],
    suspicious_chunks: List[str],
) -> List[ChunkMatch]:
    """
    Core chunk-matrix comparison engine.

    Algorithm:
      1. Embed ALL source chunks in a single batched forward pass → (N, D) matrix.
      2. Embed ALL suspicious chunks in a single batched forward pass → (M, D) matrix.
      3. Compute full (M × N) cosine similarity matrix in one vectorized call.
      4. For each suspicious chunk (row), extract the argmax over source chunks (columns).
      5. Record the best-match source index, its text, and the similarity score.

    Args:
        source_chunks:     List of string chunks from the source/reference document.
        suspicious_chunks: List of string chunks from the suspicious document.

    Returns:
        List of ChunkMatch objects — one per suspicious chunk, ordered by index.
    """
    if not source_chunks:
        raise ValueError("source_chunks cannot be empty.")
    if not suspicious_chunks:
        raise ValueError("suspicious_chunks cannot be empty.")

    # Batch embed both chunk lists — single model forward pass each
    source_embeddings = _embedding_service.get_embeddings_batch(source_chunks)    # (N, D)
    suspicious_embeddings = _embedding_service.get_embeddings_batch(suspicious_chunks)  # (M, D)

    # Full pairwise cosine similarity: (M, N)
    similarity_matrix = cosine_similarity(suspicious_embeddings, source_embeddings)

    matches: List[ChunkMatch] = []

    for susp_idx, similarity_row in enumerate(similarity_matrix):
        best_source_idx = int(np.argmax(similarity_row))
        best_score = float(round(similarity_row[best_source_idx], 4))

        matches.append(ChunkMatch(
            suspicious_chunk_index=susp_idx,
            suspicious_chunk_text=suspicious_chunks[susp_idx],
            best_match_source_index=best_source_idx,
            best_match_source_text=source_chunks[best_source_idx],
            similarity_score=best_score,
        ))

    return matches


def compute_aggregate_scores(matches: List[ChunkMatch]) -> dict:
    """
    Derives summary statistics from the per-chunk match list.

    Returns:
        dict with keys: overall_similarity (mean), max_similarity.
    """
    scores = [m.similarity_score for m in matches]
    return {
        "overall_similarity": float(round(float(np.mean(scores)), 4)),
        "max_similarity": float(round(float(np.max(scores)), 4)),
    }