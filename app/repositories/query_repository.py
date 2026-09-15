from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.query import QueryRecord
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryRepository:
    """Repository class encapsulating database operations for QueryRecord entities."""

    def __init__(self, db: Session):
        self._db = db

    def create(self, query: QueryRecord) -> QueryRecord:
        """Persist a new query and answer record."""
        self._db.add(query)
        self._db.commit()
        self._db.refresh(query)
        logger.info("Recorded query ID=%s for document_id=%s", query.id, query.document_id)
        return query

    def get_by_id(self, query_id: int) -> Optional[QueryRecord]:
        """Find a query record by its ID."""
        statement = select(QueryRecord).where(QueryRecord.id == query_id)
        return self._db.scalar(statement)

    def get_by_document_id(self, document_id: int) -> List[QueryRecord]:
        """Retrieve all queries associated with a specific document ID."""
        statement = (
            select(QueryRecord)
            .where(QueryRecord.document_id == document_id)
            .order_by(QueryRecord.created_at.desc())
        )
        return list(self._db.scalars(statement).all())
