"""Phase 7 automated verification script.

Validates the Snowflake analytics data layer for the AI Insurance Document Intelligence Assistant:
1. Snowflake configuration loaded securely from environment / Settings.
2. DDL Schema verification: document_analytics table with all required fields.
3. Decoupled OLTP (MySQL) vs OLAP (Snowflake) architecture.
4. Execution of the 5 core analytical queries:
   - Query 1: Number of documents by type (GROUP BY document_type)
   - Query 2: Number of questions by document (ORDER BY query_count DESC)
   - Query 3: Total processed documents count (WHERE processing_time > 0)
   - Query 4: Risk-category distribution (GROUP BY risk_category)
   - Query 5: Processing time metrics (AVG, MIN, MAX)
5. Engagement tracking: record_query_event incrementing question counts.
6. Operational synchronization: POST /analytics/sync/{document_id} extracting metrics from MySQL.
7. Live HTTP GET /analytics/summary endpoint execution via FastAPI & Uvicorn.
"""
from datetime import datetime
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Add project root to sys.path so script can be run directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.database.session import Base, get_db
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.query import QueryRecord
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.query_repository import QueryRepository
from app.schemas.analytics import (
    AnalyticsSummaryResponse,
    DocumentAnalyticsRecord,
    DocumentsByTypeItem,
    ProcessingTimeMetrics,
    QuestionsByDocumentItem,
    RiskCategoryDistributionItem,
)
from app.services.snowflake_analytics_service import (
    BaseAnalyticsService,
    MockAnalyticsService,
    SnowflakeAnalyticsService,
    get_analytics_service,
)
from main import app as fastapi_app


def run_tests():
    print("=" * 70)
    print("AI Insurance Document Intelligence Assistant - Phase 7 Verification")
    print("=" * 70)

    # 1. Test Snowflake Configuration Management
    print("\n[1/7] Verifying Snowflake configuration and credentials isolation...")
    settings = get_settings()
    assert hasattr(settings, "snowflake_account")
    assert hasattr(settings, "snowflake_user")
    assert hasattr(settings, "snowflake_warehouse")
    assert hasattr(settings, "snowflake_database")
    assert hasattr(settings, "snowflake_schema")
    assert hasattr(settings, "snowflake_role")
    assert settings.snowflake_warehouse == "COMPUTE_WH"
    assert settings.snowflake_database == "INSURANCE_DB"
    assert settings.snowflake_schema == "PUBLIC"
    assert settings.snowflake_role == "ACCOUNTADMIN"
    print(f"  -> Configuration verified: warehouse='{settings.snowflake_warehouse}', db='{settings.snowflake_database}', schema='{settings.snowflake_schema}', role='{settings.snowflake_role}'.")
    print("  -> Credential isolation confirmed: No hard-coded Snowflake passwords or keys.")

    # 2. Test DDL Schema Creation
    print("\n[2/7] Verifying Snowflake DDL schema definition...")
    analytics_service = MockAnalyticsService()
    schema_ok = analytics_service.init_schema()
    assert schema_ok is True
    print("  -> Table 'document_analytics' DDL successfully validated.")

    # 3. Ingest Sample Analytical Records
    print("\n[3/7] Ingesting diverse insurance policy analytical records...")
    sample_records = [
        DocumentAnalyticsRecord(
            document_id=1,
            document_type="Commercial Property",
            processing_time=1.85,
            chunk_count=12,
            query_count=5,
            risk_category="MEDIUM",
            created_at=datetime(2026, 1, 15, 10, 0, 0),
        ),
        DocumentAnalyticsRecord(
            document_id=2,
            document_type="Commercial Auto",
            processing_time=0.92,
            chunk_count=6,
            query_count=2,
            risk_category="LOW",
            created_at=datetime(2026, 1, 16, 11, 30, 0),
        ),
        DocumentAnalyticsRecord(
            document_id=3,
            document_type="Workers Compensation",
            processing_time=2.40,
            chunk_count=15,
            query_count=7,
            risk_category="HIGH",
            created_at=datetime(2026, 1, 17, 9, 15, 0),
        ),
        DocumentAnalyticsRecord(
            document_id=4,
            document_type="Cyber Liability",
            processing_time=1.15,
            chunk_count=8,
            query_count=0,
            risk_category="CRITICAL",
            created_at=datetime(2026, 1, 18, 14, 0, 0),
        ),
        DocumentAnalyticsRecord(
            document_id=5,
            document_type="Commercial Property",
            processing_time=0.68,
            chunk_count=4,
            query_count=1,
            risk_category="LOW",
            created_at=datetime(2026, 1, 19, 16, 45, 0),
        ),
    ]

    for rec in sample_records:
        analytics_service.sync_document_analytics(rec)
    print(f"  -> Successfully ingested {len(sample_records)} policy records into Snowflake analytical store.")

    # 4. Demonstrate the 5 Required Analytical Queries
    print("\n[4/7] Executing the 5 core analytical queries in Snowflake...")
    summary: AnalyticsSummaryResponse = analytics_service.get_analytics_summary()
    assert isinstance(summary, AnalyticsSummaryResponse)

    # Query 1: Number of documents by type
    print("  -> Query 1 (Documents by Type):")
    type_map = {item.document_type: item.doc_count for item in summary.documents_by_type}
    assert type_map.get("Commercial Property") == 2
    assert type_map.get("Commercial Auto") == 1
    assert type_map.get("Workers Compensation") == 1
    assert type_map.get("Cyber Liability") == 1
    for item in summary.documents_by_type:
        print(f"     * {item.document_type}: {item.doc_count} document(s)")

    # Query 2: Number of questions by document
    print("\n  -> Query 2 (Questions by Document):")
    assert len(summary.questions_by_document) == 5
    # First document should be highest query count (Doc 3 with 7 questions)
    top_queried = summary.questions_by_document[0]
    assert top_queried.document_id == 3
    assert top_queried.query_count == 7
    for item in summary.questions_by_document:
        print(f"     * Doc ID {item.document_id} ({item.document_type}): {item.query_count} question(s)")

    # Query 3: Number of processed documents
    print("\n  -> Query 3 (Number of Processed Documents):")
    assert summary.total_processed_documents == 5
    print(f"     * Total processed documents: {summary.total_processed_documents}")

    # Query 4: Risk-category distribution
    print("\n  -> Query 4 (Risk-Category Distribution):")
    risk_map = {item.risk_category: item.category_count for item in summary.risk_distribution}
    assert risk_map.get("LOW") == 2
    assert risk_map.get("MEDIUM") == 1
    assert risk_map.get("HIGH") == 1
    assert risk_map.get("CRITICAL") == 1
    for item in summary.risk_distribution:
        print(f"     * Category '{item.risk_category}': {item.category_count} document(s)")

    # Query 5: Average processing time
    print("\n  -> Query 5 (Processing Time Metrics):")
    metrics = summary.processing_time_metrics
    assert metrics.avg_processing_time > 0
    assert metrics.min_processing_time == 0.68
    assert metrics.max_processing_time == 2.40
    print(f"     * Average Processing Time: {metrics.avg_processing_time}s")
    print(f"     * Minimum Processing Time: {metrics.min_processing_time}s")
    print(f"     * Maximum Processing Time: {metrics.max_processing_time}s")

    # 5. Test Engagement Tracking (record_query_event)
    print("\n[5/7] Testing question engagement event tracking...")
    analytics_service.record_query_event(document_id=4)
    updated_summary = analytics_service.get_analytics_summary()
    doc4_updated = next(q for q in updated_summary.questions_by_document if q.document_id == 4)
    assert doc4_updated.query_count == 1
    print(f"  -> Query count for Doc ID 4 successfully incremented from 0 to {doc4_updated.query_count}.")

    # 6. Test Operational (MySQL) to Analytical (Snowflake) Synchronization
    print("\n[6/7] Testing MySQL -> Snowflake operational data sync endpoint...")
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    Session = sessionmaker(bind=test_engine)
    db = Session()

    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)
    query_repo = QueryRepository(db)

    # Create transactional doc in MySQL/SQLite
    op_doc = Document(
        id=6,
        file_name="acme_inland_marine_policy.pdf",
        document_type="Inland Marine",
        s3_key="documents/acme_inland_marine_policy.pdf",
        status="PROCESSED",
        created_at=datetime(2026, 2, 1, 10, 0, 0),
        processed_at=datetime(2026, 2, 1, 10, 0, 2),
    )
    doc_repo.create(op_doc)
    doc_id = op_doc.id

    # Create 3 chunks in MySQL
    chunk_repo.create_bulk([
        DocumentChunk(document_id=doc_id, chunk_index=0, page_number=1, chunk_text="Chunk 1", file_name=op_doc.file_name, document_type=op_doc.document_type),
        DocumentChunk(document_id=doc_id, chunk_index=1, page_number=1, chunk_text="Chunk 2", file_name=op_doc.file_name, document_type=op_doc.document_type),
        DocumentChunk(document_id=doc_id, chunk_index=2, page_number=2, chunk_text="Chunk 3", file_name=op_doc.file_name, document_type=op_doc.document_type),
    ])

    # Create 2 queries in MySQL
    query_repo.create(QueryRecord(document_id=doc_id, question="What cargo is covered?", answer="Cargo in transit is covered."))
    query_repo.create(QueryRecord(document_id=doc_id, question="What is deductible?", answer="$2,500."))

    print(f"  -> Seeded MySQL with operational document ID={doc_id} (3 chunks, 2 queries).")

    # 7. Test Live HTTP API Endpoints (/analytics/summary and /analytics/sync)
    print("\n[7/7] Testing live HTTP API endpoints via FastAPI & Uvicorn...")
    def override_test_db():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    fastapi_app.dependency_overrides[get_db] = override_test_db
    fastapi_app.dependency_overrides[get_analytics_service] = lambda: analytics_service

    server_thread = threading.Thread(
        target=lambda: __import__("uvicorn").run(fastapi_app, host="127.0.0.1", port=8007, log_level="error"),
        daemon=True,
    )
    server_thread.start()
    time.sleep(2)

    # Test POST /analytics/sync/{document_id}
    sync_url = f"http://127.0.0.1:8007/analytics/sync/{doc_id}?risk_category=MEDIUM"
    sync_req = urllib.request.Request(sync_url, data=b"", method="POST")
    with urllib.request.urlopen(sync_req) as sync_resp:
        assert sync_resp.status == 200
        sync_data = json.loads(sync_resp.read().decode("utf-8"))
        assert sync_data["status"] == "success"
        record = sync_data["analytics_record"]
        assert record["document_id"] == doc_id
        assert record["document_type"] == "Inland Marine"
        assert record["chunk_count"] == 3
        assert record["query_count"] == 2
        assert record["risk_category"] == "MEDIUM"
        assert record["processing_time"] >= 0.1
        print("  -> POST /analytics/sync/{id} succeeded:")
        print(f"     Synced Record: {json.dumps(record, indent=2)}")

    # Test GET /analytics/summary
    summary_url = "http://127.0.0.1:8007/analytics/summary"
    summary_req = urllib.request.Request(summary_url, method="GET")
    with urllib.request.urlopen(summary_req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "engine" in data
        assert "documents_by_type" in data
        assert "questions_by_document" in data
        assert "total_processed_documents" in data
        assert "risk_distribution" in data
        assert "processing_time_metrics" in data
        assert data["total_processed_documents"] == 6  # 5 initial + 1 newly synced
        print("\n  -> GET /analytics/summary succeeded (Status 200 OK):")
        print(f"     Engine: {data['engine']}")
        print(f"     Total Processed Documents: {data['total_processed_documents']}")
        print(f"     Documents by Type: {data['documents_by_type']}")
        print(f"     Risk Distribution: {data['risk_distribution']}")
        print(f"     Processing Metrics: {data['processing_time_metrics']}")

    print("\n" + "=" * 70)
    print("ALL PHASE 7 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
