# app/services/embedding_service.py
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self):
        # Load SBERT model once when the class initializes
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text: str):
        """Encodes a single text block into a list of floats."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def get_embeddings_batch(self, texts: list) -> list:
        """Batch-encode a list of strings. Returns a 2D numpy array (N, D)."""
        if not texts:
            return []
        return self.model.encode(texts, convert_to_numpy=True)
