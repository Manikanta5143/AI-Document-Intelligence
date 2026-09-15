from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.query_repository import QueryRepository
from app.schemas.analytics import AnalyticsSummaryResponse, DocumentAnalyticsRecord
from app.services.snowflake_analytics_service import BaseAnalyticsService, get_analytics_service
from app.utils.logger import setup_logger

logger = setup_logger("analytics_router")

router = APIRouter(prefix="/analytics", tags=["Snowflake Analytics"])


def get_document_repository(db: Session = Depends(get_db)) -> DocumentRepository:
    """Dependency provider for DocumentRepository with active DB session."""
    return DocumentRepository(db=db)


def get_chunk_repository(db: Session = Depends(get_db)) -> ChunkRepository:
    """Dependency provider for ChunkRepository with active DB session."""
    return ChunkRepository(db=db)


def get_query_repository(db: Session = Depends(get_db)) -> QueryRepository:
    """Dependency provider for QueryRepository with active DB session."""
    return QueryRepository(db=db)


@router.get(
    "/summary",
    response_model=AnalyticsSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Snowflake Analytical Summary",
    description=(
        "Executes 5 core business intelligence analytical queries across Snowflake:\n"
        "1. Number of documents by insurance type\n"
        "2. Number of questions per document\n"
        "3. Total number of processed documents\n"
        "4. Risk-category distribution\n"
        "5. Average, min, and max processing times."
    ),
)
def get_analytics_summary(
    analytics_service: BaseAnalyticsService = Depends(get_analytics_service),
) -> AnalyticsSummaryResponse:
    """Execute analytical queries and return aggregated OLAP reporting metrics."""
    return analytics_service.get_analytics_summary()


@router.post(
    "/sync/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Sync Operational Document Metrics to Snowflake",
    description=(
        "Collects operational metadata from MySQL (chunk count, query count, processing duration) "
        "and syncs an analytical record into Snowflake."
    ),
)
def sync_document_to_snowflake(
    document_id: int,
    risk_category: str = "LOW",
    doc_repo: DocumentRepository = Depends(get_document_repository),
    chunk_repo: ChunkRepository = Depends(get_chunk_repository),
    query_repo: QueryRepository = Depends(get_query_repository),
    analytics_service: BaseAnalyticsService = Depends(get_analytics_service),
) -> Dict[str, Any]:
    """Extract operational document metrics from MySQL and push to Snowflake analytical warehouse."""
    doc = doc_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} was not found in MySQL.",
        )

    chunks = chunk_repo.get_by_document_id(document_id)
    queries = query_repo.get_by_document_id(document_id)

    # Compute processing time in seconds if timestamps exist
    processing_time = 1.0
    if doc.processed_at and doc.created_at:
        delta = (doc.processed_at - doc.created_at).total_seconds()
        processing_time = max(round(delta, 2), 0.1)

    record = DocumentAnalyticsRecord(
        document_id=doc.id,
        document_type=doc.document_type,
        processing_time=processing_time,
        chunk_count=len(chunks),
        query_count=len(queries),
        risk_category=risk_category.upper(),
        created_at=doc.created_at,
    )

    analytics_service.sync_document_analytics(record)
    logger.info("Successfully synced document ID %d to Snowflake analytics.", document_id)

    return {
        "status": "success",
        "message": f"Document ID {document_id} synced to Snowflake analytics successfully.",
        "analytics_record": record.model_dump(),
    }
