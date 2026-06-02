from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model

    if _model is None:
        print("Loading SBERT model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    return _model
