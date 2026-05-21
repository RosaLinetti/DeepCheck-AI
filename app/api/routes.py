from fastapi import APIRouter

from app.api.schemas import AnalyzeRequest, AnalyzeResponse
from app.core.pipeline import process_text
from app.core.embedder import generate_embedding
import numpy as np

router = APIRouter()


def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    return float(
        np.dot(v1, v2) /
        (np.linalg.norm(v1) * np.linalg.norm(v2))
    )


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/")
async def root():
    return {"message": "DeepCheck API is running"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest):

    # PERSON 1 PIPELINE
    result1 = process_text(payload.text1)
    result2 = process_text(payload.text2)

    # take first chunk embedding (simple version)
    emb1 = result1["embeddings"][0]
    emb2 = result2["embeddings"][0]

    # PERSON 3 similarity logic
    score = cosine_similarity(emb1, emb2)

    return AnalyzeResponse(
        similarity_score=round(score * 100, 2),
        message="Semantic similarity computed successfully"
    )