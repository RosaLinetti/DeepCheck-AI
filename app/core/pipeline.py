from app.core.text_cleaning import clean_text
from app.core.chunking import split_sentences, sliding_window
from app.core.embedder import generate_embedding


def process_text(text: str):
    """
    FULL PERSON 1 PIPELINE:
    text → clean → sentence split → sliding window → embeddings
    """

    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)
    chunks = sliding_window(sentences)

    embeddings = [generate_embedding(chunk) for chunk in chunks]

    return {
        "chunks": chunks,
        "embeddings": embeddings
    }