from datetime import datetime
from typing import List, Optional
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.document import Document
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentRepository:
    """Repository class encapsulating database operations for the Document entity."""

    def __init__(self, db: Session):
        self._db = db

    def create(self, document: Document) -> Document:
        """Persist a new document record.

        Args:
            document: Uncommitted Document model instance.

        Returns:
            Committed Document instance with populated ID and defaults.
        """
        self._db.add(document)
        self._db.commit()
        self._db.refresh(document)
        logger.info("Created document ID=%s (file_name='%s')", document.id, document.file_name)
        return document

    def get_by_id(self, document_id: int) -> Optional[Document]:
        """Find a document by primary key ID."""
        statement = select(Document).where(Document.id == document_id)
        return self._db.scalar(statement)

    def get_all(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """Retrieve paginated document records."""
        statement = select(Document).order_by(Document.created_at.desc()).offset(skip).limit(limit)
        return list(self._db.scalars(statement).all())

    def count_all(self) -> int:
        """Count total document records in the database."""
        statement = select(func.count()).select_from(Document)
        return self._db.scalar(statement) or 0

    def update_status(
        self,
        document_id: int,
        status: str,
        processed_at: Optional[datetime] = None,
    ) -> Optional[Document]:
        """Update the processing status of a document."""
        document = self.get_by_id(document_id)
        if document:
            document.status = status
            if processed_at:
                document.processed_at = processed_at
            self._db.commit()
            self._db.refresh(document)
            logger.info("Updated document ID=%s status to '%s'", document_id, status)
        return document
