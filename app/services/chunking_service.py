"""
Chunking Service — DeepCheck-AI
Provides two strategies:
  - SENTENCE: tokenizes text into individual sentences via regex
  - SLIDING_WINDOW: produces overlapping fixed-size token windows
"""

import re
from enum import Enum
from typing import List


class ChunkStrategy(str, Enum):
    SENTENCE = "sentence"
    SLIDING_WINDOW = "sliding_window"


def _tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation-aware regex."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 2]


def chunk_by_sentence(text: str) -> List[str]:
    """
    Returns a list of sentence-level string chunks.
    Each chunk maps to one index in the downstream comparison matrix.
    """
    return _tokenize_sentences(text)


def chunk_by_sliding_window(
    text: str,
    window_size: int = 30,
    overlap: int = 10
) -> List[str]:
    """
    Returns overlapping token-window chunks as reconstructed strings.
    Preserves original casing and punctuation for high-quality UI highlighting.
    """
    if overlap >= window_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than window_size ({window_size})."
        )

    words = text.strip().split()
    step = window_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        window = words[start : start + window_size]
        if len(window) < 5:  # discard micro-fragments at tail
            break
        chunks.append(" ".join(window))

    return chunks


def get_chunks(
    text: str,
    strategy: ChunkStrategy,
    window_size: int = 30,
    overlap: int = 10
) -> List[str]:
    """
    Unified dispatcher — routes to the correct chunker based on strategy enum.
    """
    if strategy == ChunkStrategy.SENTENCE:
        return chunk_by_sentence(text)
    elif strategy == ChunkStrategy.SLIDING_WINDOW:
        return chunk_by_sliding_window(text, window_size=window_size, overlap=overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")