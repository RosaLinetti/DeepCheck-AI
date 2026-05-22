import re

def split_sentences(text: str):
    return re.split(r'(?<=[.!?])\s+', text.strip())


def sliding_window(sentences, window_size=3, overlap=1):
    chunks = []
    i = 0

    while i < len(sentences):
        chunk = sentences[i:i + window_size]
        chunks.append(" ".join(chunk))
        i += window_size - overlap

    return chunks