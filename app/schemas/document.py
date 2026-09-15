from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentBase(BaseModel):
    """Base document fields."""

    file_name: str = Field(..., min_length=1, max_length=255, description="Name of the insurance document file")
    document_type: str = Field(..., min_length=1, max_length=100, description="Type of document (e.g. Policy, Claim, Endorsement)")


class DocumentCreate(DocumentBase):
    """Schema for creating document metadata."""

    s3_key: Optional[str] = Field(None, max_length=500, description="Optional S3 object key if pre-assigned")

    model_config = {
        "json_schema_extra": {
            "example": {
                "file_name": "auto_policy_2026.pdf",
                "document_type": "Auto Policy",
                "s3_key": None,
            }
        }
    }


class DocumentResponse(DocumentBase):
    """Schema for document details response."""

    id: int = Field(..., description="Unique document ID in database")
    s3_key: Optional[str] = Field(None, description="S3 storage key")
    status: str = Field(..., description="Current processing status (e.g. PENDING, PROCESSED)")
    created_at: datetime = Field(..., description="Timestamp when the document was registered")
    processed_at: Optional[datetime] = Field(None, description="Timestamp when document processing finished")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": 1,
                "file_name": "auto_policy_2026.pdf",
                "document_type": "Auto Policy",
                "s3_key": None,
                "status": "PENDING",
                "created_at": "2026-09-14T10:00:00Z",
                "processed_at": None,
            }
        },
    }


class DocumentListResponse(BaseModel):
    """Schema for paginated list of documents."""

    total: int = Field(..., description="Total number of documents returned")
    documents: List[DocumentResponse] = Field(..., description="List of document records")


class DocumentUploadResponse(BaseModel):
    """Schema returned after a successful document upload and chunking."""

    document_id: int = Field(..., description="Unique document ID in database")
    file_name: str = Field(..., description="Original file name")
    document_type: str = Field(..., description="Insurance document category")
    s3_key: str = Field(..., description="Destination S3 object key")
    status: str = Field(..., description="Processing status (e.g. PROCESSED)")
    total_pages: int = Field(..., description="Total number of pages extracted from PDF")
    total_chunks: int = Field(..., description="Total text chunks generated and saved")
    chunks: List["ChunkResponse"] = Field(..., description="List of chunk metadata records")

    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": 1,
                "file_name": "homeowners_policy.pdf",
                "document_type": "Homeowners Policy",
                "s3_key": "documents/1a2b3c4d5e6f_homeowners_policy.pdf",
                "status": "PROCESSED",
                "total_pages": 3,
                "total_chunks": 5,
                "chunks": [],
            }
        }
    }


# Resolve forward references
from app.schemas.chunk import ChunkResponse  # noqa: E402
DocumentUploadResponse.model_rebuild()

