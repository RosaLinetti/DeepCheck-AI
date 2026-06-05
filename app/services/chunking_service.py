"""
Text chunking utilities used before plagiarism checking.

- Sentence: splits text into sentences.
- Sliding window: creates overlapping word chunks.
"""

import re
from enum import Enum
from typing import List


class ChunkStrategy(str, Enum):
    SENTENCE = "sentence"
    SLIDING_WINDOW = "sliding_window"


def _tokenize_sentences(text: str) -> List[str]:
    """Use regex to separate text into sentences."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 2]


def chunk_by_sentence(text: str) -> List[str]:
    """Break text into sentence-level chunks for comparison."""
    return _tokenize_sentences(text)


def chunk_by_sliding_window(
    text: str,
    window_size: int = 30,
    overlap: int = 10
) -> List[str]:
    """
    Create overlapping word chunks using a sliding window.
    Keeps the original text formatting intact.
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
    Return chunks based on the selected strategy.
    """
    if strategy == ChunkStrategy.SENTENCE:
        return chunk_by_sentence(text)
    elif strategy == ChunkStrategy.SLIDING_WINDOW:
        return chunk_by_sliding_window(text, window_size=window_size, overlap=overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
