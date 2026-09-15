from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


class QueryRecord(Base):
    """SQLAlchemy model representing a Q&A interaction regarding a document."""

    __tablename__ = "queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    document_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship back to parent document
    document: Mapped[Optional["Document"]] = relationship("Document", back_populates="queries")  # noqa: F821

    def __repr__(self) -> str:
        return f"<QueryRecord(id={self.id}, doc_id={self.document_id}, created_at={self.created_at})>"
