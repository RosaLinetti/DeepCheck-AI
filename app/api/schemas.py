from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):

    text1: str = Field(
        ...,
        min_length=1,
        description="First text input."
    )

    text2: str = Field(
        ...,
        min_length=1,
        description="Second text input."
    )


class AnalyzeResponse(BaseModel):

    similarity_score: float

    message: str