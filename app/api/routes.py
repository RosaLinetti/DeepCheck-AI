from fastapi import APIRouter

from app.api.schemas import AnalyzeRequest, AnalyzeResponse

from app.services.embedding_service import generate_embedding
from app.services.similarity_service import calculate_similarity

router = APIRouter()


@router.get("/health")
async def health():

    return {"status": "ok"}


@router.get("/")
async def root():

    return {"message": "DeepCheck API is running"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest):

    embedding1 = generate_embedding(payload.text1)

    embedding2 = generate_embedding(payload.text2)

    similarity_score = calculate_similarity(
        embedding1,
        embedding2
    )

    similarity_percentage = round(similarity_score * 100, 2)

    return AnalyzeResponse(
        similarity_score=similarity_percentage,
        message="Semantic similarity analysis completed."
    )