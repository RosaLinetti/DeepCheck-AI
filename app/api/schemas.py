from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text1: str = Field(..., min_length=1)
    text2: str = Field(..., min_length=1)


class AnalyzeResponse(BaseModel):
    similarity_score: float
    message: str