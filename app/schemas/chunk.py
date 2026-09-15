from typing import Optional
from pydantic import BaseModel, Field


class ChunkResponse(BaseModel):
    """Schema representing an extracted and stored document chunk."""

    chunk_id: int = Field(..., description="Unique database ID of the chunk")
    document_id: int = Field(..., description="ID of the parent document")
    chunk_index: int = Field(..., description="Sequential index of the chunk within the document")
    page_number: int = Field(..., description="Source page number (1-indexed)")
    text: str = Field(..., description="Cleaned chunk text")
    file_name: Optional[str] = Field(None, description="Original file name")
    document_type: Optional[str] = Field(None, description="Document type")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "chunk_id": 1,
                "document_id": 1,
                "chunk_index": 0,
                "page_number": 1,
                "text": "Coverage A - Dwelling: $500,000 limit with $1,000 deductible.",
                "file_name": "homeowners_policy.pdf",
                "document_type": "Homeowners Policy",
            }
        },
    }
