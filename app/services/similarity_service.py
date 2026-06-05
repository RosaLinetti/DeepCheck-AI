"""
Similarity Service — DeepCheck-AI
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from app.services.embedding_service import embedding_service as _embedding_service
from app.services.classifier_service import classify_chunk

def _compute_multi_feature_score(
    susp_text: str,
    src_text: str,
    cosine_score: float,
    w_cosine: float = 0.50,
    w_length: float = 0.25,
    w_lexical: float = 0.25,
) -> float:
    """
    Combine cosine similarity with:
    - semantic similarity (SBERT cosine)
    - length similarity
    - lexical overlap (Jaccard)
    into a single calibrated score.
    """
    cosine = cosine_score

    # Length similarity
    len_susp = max(len(susp_text.split()), 1)
    len_src = max(len(src_text.split()), 1)

    length_ratio = min(len_susp, len_src) / max(len_susp, len_src)

    # Lexical overlap
    words_susp = set(susp_text.lower().split())
    words_src = set(src_text.lower().split())

    intersection = words_susp & words_src
    union = words_susp | words_src

    lexical_overlap = len(intersection) / len(union) if union else 0.0

    final_score = (
        (w_cosine * cosine)
        + (w_length * length_ratio)
        + (w_lexical * lexical_overlap)
    )

    # Summary / synthesis adjustment
    if lexical_overlap < 0.12 and length_ratio < 0.35:
        final_score *= 0.50

    return float(round(min(1.0, max(0.0, final_score)), 4))


def _get_adaptive_verdict(
    score: float,
    total_chunks: int,
) -> str:
    """
    Thresholds change slightly depending on document size.
    """
    if total_chunks <= 3:
        high, mid = 0.78, 0.55
    elif total_chunks <= 8:
        high, mid = 0.75, 0.52
    else:
        high, mid = 0.72, 0.50

    if score >= high:
        return "plagiarised"
    elif score >= mid:
        return "suspicious"
    else:
        return "original"


def compute_similarity(text1: str, text2: str) -> float:
    """
    Legacy whole-text cosine similarity.
    """
    vec1 = _embedding_service.generate_embedding(text1)
    vec2 = _embedding_service.generate_embedding(text2)

    score = cosine_similarity([vec1], [vec2])[0][0]

    return float(round(score, 4))


def compute_chunk_similarity_matrix(
    source_chunks: List[str],
    suspicious_chunks: List[str],
) -> List[Dict[str, Any]]:
    """
    Compare each suspicious chunk against source chunks
    and find the best matching source chunk.
    """
    if not source_chunks:
        raise ValueError("source_chunks cannot be empty.")

    if not suspicious_chunks:
        raise ValueError("suspicious_chunks cannot be empty.")

    source_embeddings = _embedding_service.get_embeddings_batch(source_chunks)
    suspicious_embeddings = _embedding_service.get_embeddings_batch(
        suspicious_chunks
    )

    similarity_matrix = cosine_similarity(
        suspicious_embeddings,
        source_embeddings,
    )

    matches = []

    for susp_idx, similarity_row in enumerate(similarity_matrix):
        best_source_idx = int(np.argmax(similarity_row))
        best_score = float(round(float(similarity_row[best_source_idx]), 4))

        susp_text = suspicious_chunks[susp_idx]
        src_text = source_chunks[best_source_idx]

        # Calculate reusable features once
        length_ratio = (
            min(len(susp_text.split()), len(src_text.split()))
            / max(len(susp_text.split()), len(src_text.split()))
        )

        words_susp = set(susp_text.lower().split())
        words_src = set(src_text.lower().split())

        lexical_overlap = (
            len(words_susp & words_src)
            / len(words_susp | words_src)
            if (words_susp | words_src)
            else 0.0
        )

        final_score = _compute_multi_feature_score(
            susp_text=susp_text,
            src_text=src_text,
            cosine_score=best_score,
        )

        verdict, confidence = classify_chunk(
            cosine_score=best_score,
            length_ratio=length_ratio,
            lexical_overlap=lexical_overlap,
        )

        # Keep score consistent with classifier verdict
        if verdict == "original":
            final_score = min(final_score, 0.35)

        matches.append(
            {
                "suspicious_chunk_index": int(susp_idx),
                "suspicious_chunk_text": susp_text,
                "best_match_source_index": int(best_source_idx),
                "best_match_source_text": src_text,
                "similarity_score": final_score,
                "verdict": verdict,
                "confidence": confidence,
            }
        )

    return matches


def compute_aggregate_scores(matches: List[Dict[str, Any]]) -> dict:
    """
    Mean and max similarity across all chunk matches.
    """
    scores = [m["similarity_score"] for m in matches]

    if not scores:
        return {
            "overall_similarity": 0.0,
            "max_similarity": 0.0,
        }

    return {
        "overall_similarity": float(round(float(np.mean(scores)), 4)),
        "max_similarity": float(round(float(np.max(scores)), 4)),
    }