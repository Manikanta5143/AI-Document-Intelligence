from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentAnalyticsRecord(BaseModel):
    """Data transfer model for inserting/updating an analytical document record in Snowflake."""

    document_id: int = Field(..., description="Operational document ID from MySQL")
    document_type: str = Field(..., description="Insurance document classification (e.g. Commercial Property, Auto)")
    processing_time: float = Field(..., ge=0.0, description="Processing and chunking duration in seconds")
    chunk_count: int = Field(..., ge=0, description="Total number of chunks generated")
    query_count: int = Field(default=0, ge=0, description="Number of questions asked against this document")
    risk_category: str = Field(default="LOW", description="Risk assessment category: LOW, MEDIUM, HIGH, CRITICAL")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": 1,
                "document_type": "Commercial Property",
                "processing_time": 1.45,
                "chunk_count": 8,
                "query_count": 3,
                "risk_category": "MEDIUM",
            }
        }
    }


class DocumentsByTypeItem(BaseModel):
    """Analytical distribution of documents by policy type."""

    document_type: str = Field(..., description="Type or category of the insurance policy")
    doc_count: int = Field(..., ge=0, description="Number of documents in this category")


class QuestionsByDocumentItem(BaseModel):
    """Analytical ranking of question volume per document."""

    document_id: int = Field(..., description="Document identifier")
    document_type: str = Field(..., description="Document category")
    query_count: int = Field(..., ge=0, description="Number of questions answered")


class RiskCategoryDistributionItem(BaseModel):
    """Analytical distribution of risk classifications across policies."""

    risk_category: str = Field(..., description="Risk tier: LOW, MEDIUM, HIGH, CRITICAL")
    category_count: int = Field(..., ge=0, description="Total documents matching this risk tier")


class ProcessingTimeMetrics(BaseModel):
    """Aggregated processing performance metrics in seconds."""

    avg_processing_time: float = Field(..., description="Average processing duration in seconds")
    min_processing_time: float = Field(..., description="Minimum processing duration in seconds")
    max_processing_time: float = Field(..., description="Maximum processing duration in seconds")


class AnalyticsSummaryResponse(BaseModel):
    """Consolidated Snowflake analytical report executing the 5 core business intelligence queries."""

    engine: str = Field(
        default="Snowflake OLAP",
        description="Analytics execution engine (Snowflake Live or Mock Engine)",
    )
    total_processed_documents: int = Field(..., description="Query 3: Total count of processed documents")
    documents_by_type: List[DocumentsByTypeItem] = Field(
        default_factory=list,
        description="Query 1: Distribution of documents by insurance policy type",
    )
    questions_by_document: List[QuestionsByDocumentItem] = Field(
        default_factory=list,
        description="Query 2: Question volume and engagement per document",
    )
    risk_distribution: List[RiskCategoryDistributionItem] = Field(
        default_factory=list,
        description="Query 4: Risk classification distribution across policyholders",
    )
    processing_time_metrics: ProcessingTimeMetrics = Field(
        ...,
        description="Query 5: Average, min, and max document processing latencies",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "engine": "Snowflake OLAP",
                "total_processed_documents": 12,
                "documents_by_type": [
                    {"document_type": "Commercial Property", "doc_count": 6},
                    {"document_type": "Commercial Auto", "doc_count": 4},
                    {"document_type": "Workers Compensation", "doc_count": 2},
                ],
                "questions_by_document": [
                    {"document_id": 1, "document_type": "Commercial Property", "query_count": 8},
                    {"document_id": 2, "document_type": "Commercial Auto", "query_count": 3},
                ],
                "risk_distribution": [
                    {"risk_category": "LOW", "category_count": 7},
                    {"risk_category": "MEDIUM", "category_count": 3},
                    {"risk_category": "HIGH", "category_count": 2},
                ],
                "processing_time_metrics": {
                    "avg_processing_time": 1.25,
                    "min_processing_time": 0.42,
                    "max_processing_time": 2.89,
                },
            }
        }
    }
