from typing import List
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.chunk import DocumentChunk
from app.utils.logger import setup_logger

logger = setup_logger("chunk_repository")


class ChunkRepository:
    """Repository class encapsulating database operations for the DocumentChunk entity."""

    def __init__(self, db: Session):
        self._db = db

    def create_bulk(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Persist multiple document chunks in a single transaction.

        Args:
            chunks: List of uncommitted DocumentChunk model instances.

        Returns:
            List[DocumentChunk]: Committed chunks with populated primary keys.
        """
        if not chunks:
            return []

        self._db.add_all(chunks)
        self._db.commit()
        for chunk in chunks:
            self._db.refresh(chunk)

        logger.info("Persisted %d chunks to database.", len(chunks))
        return chunks

    def get_by_document_id(self, document_id: int) -> List[DocumentChunk]:
        """Retrieve all chunks associated with a document ordered by chunk index."""
        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
        return list(self._db.scalars(statement).all())

    def count_by_document_id(self, document_id: int) -> int:
        """Count total chunks for a document."""
        statement = (
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
        )
        return self._db.scalar(statement) or 0
