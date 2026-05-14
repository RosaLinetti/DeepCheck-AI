from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to analyze (placeholder).")


class AnalyzeResponse(BaseModel):
    status: str = Field(..., description="Processing status.")
    message: str = Field(..., description="Dummy payload until real analysis is wired.")
