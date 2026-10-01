from pydantic import BaseModel, Field
from typing import List


class UploadResponse(BaseModel):
    doc_id: str = Field(..., description="ID assigned to the uploaded document")
    filename: str
    chunk_count: int


class QueryRequest(BaseModel):
    doc_id: str = Field(..., description="ID of the previously uploaded document")
    question: str = Field(..., min_length=3, max_length=1000)


class SourceChunk(BaseModel):
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class ErrorResponse(BaseModel):
    detail: str