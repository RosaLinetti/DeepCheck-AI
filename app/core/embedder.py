from app.core.model_loader import get_model

def generate_embedding(text: str):
    if not text or not text.strip():
        return []

    model = get_model()
    return model.encode(text).tolist()
