from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class QueryBase(BaseModel):
    """Base query schema."""

    question: str = Field(..., min_length=1, description="Question asked about the document")


class QueryCreate(QueryBase):
    """Schema for creating a query."""

    document_id: Optional[int] = Field(None, description="Associated document ID")


class QueryResponse(QueryBase):
    """Schema for query and answer response."""

    id: int = Field(..., description="Query ID")
    document_id: Optional[int] = Field(None, description="Associated document ID")
    answer: str = Field(..., description="Generated answer")
    created_at: datetime = Field(..., description="Timestamp when query was recorded")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": 1,
                "document_id": 1,
                "question": "What is the deductible for collision coverage?",
                "answer": "The collision coverage deductible is $500.",
                "created_at": "2026-09-14T10:05:00Z",
            }
        },
    }
