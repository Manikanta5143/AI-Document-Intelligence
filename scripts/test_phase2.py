"""Phase 2 verification script.

Tests all Phase 2 components (SQLAlchemy models, repositories, service layer,
and schemas) using an isolated in-memory SQLite database.
"""
import sys
from pathlib import Path

# Add project root to sys.path so script can be run directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

from app.database.session import Base
from app.models.document import Document
from app.models.chunk import DocumentChunk
from app.models.query import QueryRecord
from app.repositories.document_repository import DocumentRepository
from app.repositories.query_repository import QueryRepository
from app.services.document_service import DocumentService
from app.schemas.document import DocumentCreate, DocumentResponse, DocumentListResponse


def run_tests():
    print("=" * 60)
    print("AI Insurance Document Intelligence Assistant - Phase 2 Tests")
    print("=" * 60)

    # 1. Setup isolated in-memory SQLite database
    print("\n[1/5] Initializing in-memory test database and tables...")
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    print("  -> Tables created successfully in SQLite memory.")

    # 2. Test Repository and Service: Create Document Metadata
    print("\n[2/5] Testing Document Creation (Service & Repository)...")
    repo = DocumentRepository(db)
    service = DocumentService(repo)

    doc_in = DocumentCreate(
        file_name="commercial_property_policy_2026.pdf",
        document_type="Commercial Property",
        s3_key="uploads/commercial_property_policy_2026.pdf",
    )
    doc_res = service.create_document_metadata(doc_in)
    assert doc_res.id == 1, f"Expected ID 1, got {doc_res.id}"
    assert doc_res.file_name == "commercial_property_policy_2026.pdf"
    assert doc_res.status == "PENDING"
    print(f"  -> Created document ID={doc_res.id} with status='{doc_res.status}'")

    # 3. Test Retrieval and 404 handling
    print("\n[3/5] Testing Document Retrieval & Error Handling...")
    fetched = service.get_document_by_id(1)
    assert fetched.id == 1
    assert fetched.document_type == "Commercial Property"
    print(f"  -> Successfully retrieved document ID={fetched.id}")

    try:
        service.get_document_by_id(9999)
        raise AssertionError("Expected HTTPException for missing document")
    except HTTPException as e:
        assert e.status_code == 404
        print(f"  -> Non-existent ID 9999 correctly raised HTTP 404: {e.detail}")

    # 4. Test Listing & Pagination
    print("\n[4/5] Testing List and Pagination...")
    # Add a second document
    doc_in_2 = DocumentCreate(
        file_name="auto_fleet_endorsement.pdf",
        document_type="Auto Endorsement",
    )
    service.create_document_metadata(doc_in_2)

    doc_list = service.list_documents(skip=0, limit=10)
    assert doc_list.total == 2, f"Expected total 2, got {doc_list.total}"
    assert len(doc_list.documents) == 2
    print(f"  -> Listed {len(doc_list.documents)} documents (total: {doc_list.total})")

    # 5. Test Relational Models (Chunks & Query Records)
    print("\n[5/5] Testing Relational Tables (DocumentChunks & QueryRecords)...")
    chunk_1 = DocumentChunk(
        document_id=doc_res.id,
        chunk_index=0,
        page_number=1,
        chunk_text="Coverage A - Building Property: $1,000,000 limit with $5,000 deductible.",
    )
    chunk_2 = DocumentChunk(
        document_id=doc_res.id,
        chunk_index=1,
        page_number=2,
        chunk_text="Exclusions: Flood, Earthquake, and Normal Wear and Tear.",
    )
    db.add_all([chunk_1, chunk_2])
    db.commit()

    q_repo = QueryRepository(db)
    query_rec = QueryRecord(
        document_id=doc_res.id,
        question="What is the building property coverage limit?",
        answer="Building Property limit is $1,000,000 with a $5,000 deductible.",
    )
    q_repo.create(query_rec)

    queries = q_repo.get_by_document_id(doc_res.id)
    assert len(queries) == 1
    assert queries[0].question == "What is the building property coverage limit?"
    print(f"  -> Document ID={doc_res.id} has {len(queries)} query record and 2 chunks attached.")

    # Test status update
    updated_doc = repo.update_status(doc_res.id, "PROCESSED", processed_at=datetime.utcnow())
    assert updated_doc.status == "PROCESSED"
    assert updated_doc.processed_at is not None
    print(f"  -> Updated status to '{updated_doc.status}' with timestamp {updated_doc.processed_at}")

    db.close()
    print("\n" + "=" * 60)
    print("ALL PHASE 2 UNIT & INTEGRATION TESTS PASSED (100% SUCCESS)!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
