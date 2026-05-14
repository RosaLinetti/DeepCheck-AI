from fastapi import APIRouter

from app.api.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    return AnalyzeResponse(
        status="accepted",
        message=f"Dummy analyze for {len(payload.text)} characters.",
    )
