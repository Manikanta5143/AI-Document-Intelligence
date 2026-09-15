from typing import List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request payload for semantic chunk retrieval."""

    query: str = Field(..., min_length=1, max_length=2000, description="Question or search phrase")
    document_id: Optional[int] = Field(None, description="Filter results to a specific document ID")
    document_type: Optional[str] = Field(None, description="Filter results to a specific document category (e.g. Policy, Claim)")
    top_k: int = Field(5, ge=1, le=20, description="Number of most relevant chunks to return")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the deductible for building property damage?",
                "document_id": 1,
                "top_k": 5,
            }
        }
    }


class SearchResultItem(BaseModel):
    """Metadata and content for a single retrieved chunk."""

    chunk_id: int = Field(..., description="Unique database ID of the chunk")
    document_id: int = Field(..., description="ID of the parent document")
    document_name: str = Field(..., description="Original file name of the document")
    document_type: str = Field(..., description="Insurance document category")
    page_number: int = Field(..., description="Page number where the chunk is located (1-indexed)")
    text: str = Field(..., description="Cleaned chunk text content")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0, higher is more similar)")


class SearchResponse(BaseModel):
    """Response returned for semantic similarity search."""

    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total matching chunks returned")
    results: List[SearchResultItem] = Field(..., description="List of top-k retrieved chunks ranked by relevance")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the deductible for building property damage?",
                "total_results": 1,
                "results": [
                    {
                        "chunk_id": 2,
                        "document_id": 1,
                        "document_name": "commercial_policy.pdf",
                        "document_type": "Commercial Property",
                        "page_number": 2,
                        "text": "Standard Deductible: $5,000 per occurrence. Windstorm/Hail Deductible: 2%.",
                        "similarity_score": 0.884,
                    }
                ],
            }
        }
    }
