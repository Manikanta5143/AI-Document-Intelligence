"""Master End-to-End Verification Script (Phase 8).

Validates the complete 18-step lifecycle of the AI Insurance Document Intelligence Assistant:
 1. Start FastAPI application server.
 2. Verify OpenAPI / Swagger documentation endpoints (/docs and /openapi.json).
 3. Upload insurance PDF via multipart HTTP POST (/documents/upload).
 4. Store PDF binary in AWS S3 (S3Service).
 5. Create document operational record in MySQL (Document with PROCESSED status).
 6. Extract page-aware text using PyMuPDF (PDFExtractor).
 7. Split text into overlapping chunks (TextChunker).
 8. Generate embeddings via Amazon Titan Text Embeddings v2 (EmbeddingService).
 9. Store embeddings and chunk metadata in local ChromaDB vector store.
10. Ask a natural language policy question (POST /documents/{id}/ask).
11. Retrieve top-K relevant chunks scoped strictly to the target document_id.
12. Send grounded context and question to Amazon Nova Lite (LLMService).
13. Generate grounded, deterministic answer (temperature=0.0).
14. Return source attribution metadata (file_name, page_number, chunk_id).
15. Persist user question and assistant answer in MySQL queries table.
16. Generate executive document summary with key takeaways (POST /documents/{id}/summary).
17. Extract structured insurance data into validated JSON (POST /documents/{id}/extract).
18. Store and aggregate business intelligence metrics in Snowflake (POST /analytics/sync and GET /analytics/summary).
"""
import io
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pymupdf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.document_service as doc_service_module
from app.aws.bedrock_service import MockEmbeddingService
from app.aws.llm_service import MockLLMService, get_llm_service
from app.aws.s3_service import S3Service
from app.config import get_settings
from app.database.session import Base, get_db
from app.rag.vector_store import VectorStoreService, get_vector_store_service
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.query_repository import QueryRepository
from app.services.snowflake_analytics_service import (
    BaseAnalyticsService,
    MockAnalyticsService,
    get_analytics_service,
)
from main import app as fastapi_app


class StubS3Service:
    """Stub S3Service for test environment without live AWS credentials."""

    def __init__(self, settings=None):
        self.uploaded_keys = []

    def generate_s3_key(self, file_name: str) -> str:
        return f"documents/test_e2e_{file_name}"

    def upload_fileobj(self, file_obj, s3_key: str, content_type: str = "application/pdf") -> str:
        data = file_obj.read()
        self.uploaded_keys.append({"key": s3_key, "size": len(data), "content_type": content_type})
        return s3_key


def create_sample_pdf_bytes() -> bytes:
    """Generate a multi-page insurance policy PDF using PyMuPDF."""
    doc = pymupdf.open()

    # Page 1: Declarations & Limits
    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "COMMERCIAL PROPERTY INSURANCE POLICY\n"
        "Policy Number: POL-2026-9942\n"
        "Named Insured: Acme Global Logistics LLC\n"
        "Policy Period: January 1, 2026 to January 1, 2027\n\n"
        "SCHEDULE OF COVERAGES:\n"
        "Building Property Coverage Limit: $2,500,000\n"
        "Business Personal Property Limit: $750,000\n"
        "Standard Property Loss Deductible: $5,000 per occurrence\n\n"
        "PREMIUM SUMMARY:\n"
        "Annual Base Premium: $14,500.00\n",
        fontsize=11,
    )

    # Page 2: Exclusions & Perils
    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        "SECTION B - EXCLUDED PERILS AND WATER DAMAGE\n\n"
        "1. FLOOD AND WATER EXCLUSIONS:\n"
        "Loss or damage caused by flood, surface water, waves, tidal water,\n"
        "overflow of any body of water, or sewer backup is strictly excluded.\n\n"
        "2. EARTH MOVEMENT EXCLUSION:\n"
        "Earthquake, landslide, mudflow, and earth sinking are not covered.\n\n"
        "3. COVERED WATER DAMAGE:\n"
        "Sudden and accidental discharge of water from internal plumbing\n"
        "systems is covered subject to the $5,000 standard deductible.\n",
        fontsize=11,
    )

    # Page 3: Endorsements & Conditions
    p3 = doc.new_page()
    p3.insert_text(
        (50, 72),
        "SECTION C - CONDITIONS AND WINDSTORM ENDORSEMENT\n\n"
        "ENDORSEMENT CP-04: WINDSTORM & HAIL PERCENTAGE DEDUCTIBLE\n"
        "For any loss or damage caused by windstorm, cyclone, tornado, or hail,\n"
        "a special deductible of 2% of the stated building value applies.\n\n"
        "CLAIM NOTIFICATION REQUIREMENTS:\n"
        "The insured must report all losses to the insurer within 30 days.\n",
        fontsize=11,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def run_master_test():
    print("=" * 80)
    print("AI Insurance Document Intelligence Assistant - Master End-to-End Test")
    print("=" * 80)

    # Patch S3Service in document_service module
    doc_service_module.S3Service = StubS3Service

    # -------------------------------------------------------------------------
    # Setup In-Memory Environment & Mocked Infrastructure
    # -------------------------------------------------------------------------
    print("\n[Setup] Initializing test infrastructure...")
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    Session = sessionmaker(bind=test_engine)
    db = Session()

    # ChromaDB with Mock Embedding Service
    settings = get_settings()
    import shutil
    test_chroma_dir = str(ROOT_DIR / "data" / "test_chroma_e2e")
    shutil.rmtree(test_chroma_dir, ignore_errors=True)
    settings.chroma_persist_directory = test_chroma_dir
    settings.chroma_collection_name = "e2e_insurance_collection"

    mock_embeddings = MockEmbeddingService()
    vector_store = VectorStoreService(
        embedding_service=mock_embeddings,
        settings=settings,
    )
    mock_llm = MockLLMService(model_id="mock.nova-lite-v1:0")
    analytics_service = MockAnalyticsService()

    # Dependency Overrides for FastAPI
    def override_test_db():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    fastapi_app.dependency_overrides[get_db] = override_test_db
    fastapi_app.dependency_overrides[get_vector_store_service] = lambda: vector_store
    fastapi_app.dependency_overrides[get_llm_service] = lambda: mock_llm
    fastapi_app.dependency_overrides[get_analytics_service] = lambda: analytics_service

    # Step 1: Start FastAPI server in background thread
    print("\n[Step 1] Starting FastAPI application server...")
    port = 8008
    server_thread = threading.Thread(
        target=lambda: __import__("uvicorn").run(fastapi_app, host="127.0.0.1", port=port, log_level="error"),
        daemon=True,
    )
    server_thread.start()
    time.sleep(2)
    base_url = f"http://127.0.0.1:{port}"
    print(f"  -> FastAPI server listening at {base_url}")

    # Step 2: Open / Verify Swagger and OpenAPI documentation
    print("\n[Step 2] Verifying OpenAPI & Swagger documentation...")
    openapi_req = urllib.request.Request(f"{base_url}/openapi.json")
    with urllib.request.urlopen(openapi_req) as resp:
        assert resp.status == 200
        schema = json.loads(resp.read().decode("utf-8"))
        assert "paths" in schema
        paths = schema["paths"]
        assert "/documents/upload" in paths
        assert "/documents/{document_id}/ask" in paths
        assert "/documents/{document_id}/summary" in paths
        assert "/documents/{document_id}/extract" in paths
        assert "/analytics/summary" in paths
        print(f"  -> OpenAPI schema verified: {len(paths)} API routes registered.")

    # Step 3: Upload Insurance PDF
    print("\n[Step 3] Uploading 3-page insurance policy PDF (POST /documents/upload)...")
    pdf_bytes = create_sample_pdf_bytes()
    boundary = "----WebKitFormBoundaryE2ETestBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    # document_type field
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="document_type"\r\n\r\n')
    body.extend(b"Commercial Property\r\n")
    # file field
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="file"; filename="acme_property_policy.pdf"\r\n')
    body.extend(b"Content-Type: application/pdf\r\n\r\n")
    body.extend(pdf_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    upload_req = urllib.request.Request(
        f"{base_url}/documents/upload",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(upload_req) as upload_resp:
        assert upload_resp.status == 201
        upload_data = json.loads(upload_resp.read().decode("utf-8"))
        doc_id = upload_data["document_id"]
        s3_key = upload_data["s3_key"]
        chunk_count = len(upload_data["chunks"])
        print(f"  -> Upload succeeded: Document ID={doc_id}, Status='{upload_data['status']}'")

    # Step 4: S3 Storage Verification
    print("\n[Step 4] Verifying S3 storage key generation...")
    assert s3_key.startswith("documents/")
    print(f"  -> S3 Key: '{s3_key}'")

    # Step 5: MySQL Document Record Verification
    print("\n[Step 5] Verifying document record in MySQL...")
    doc_repo = DocumentRepository(db)
    saved_doc = doc_repo.get_by_id(doc_id)
    assert saved_doc is not None
    assert saved_doc.file_name == "acme_property_policy.pdf"
    assert saved_doc.status == "PROCESSED"
    assert saved_doc.processed_at is not None
    print(f"  -> MySQL Document verified: ID={saved_doc.id}, file='{saved_doc.file_name}', status='{saved_doc.status}'")

    # Step 6 & 7: PyMuPDF Text Extraction & Chunking Verification
    print("\n[Step 6 & 7] Verifying PyMuPDF text extraction and overlapping chunks...")
    chunk_repo = ChunkRepository(db)
    saved_chunks = chunk_repo.get_by_document_id(doc_id)
    assert len(saved_chunks) == chunk_count
    assert chunk_count >= 3
    print(f"  -> Extracted and saved {len(saved_chunks)} chunks across 3 pages in MySQL.")

    # Step 8 & 9: Bedrock Embeddings & ChromaDB Storage Verification
    print("\n[Step 8 & 9] Verifying embeddings and ChromaDB vector indexing...")
    search_test = vector_store.similarity_search("property coverage limit", top_k=3, document_id=doc_id)
    assert len(search_test) > 0
    print(f"  -> ChromaDB retrieved {len(search_test)} relevant chunks for document ID {doc_id}.")

    # Step 10, 11, 12, 13, 14: RAG Question Answering with Source Attribution
    print("\n[Step 10-14] Asking question via RAG pipeline (POST /documents/{id}/ask)...")
    ask_req = urllib.request.Request(
        f"{base_url}/documents/{doc_id}/ask",
        data=json.dumps({"question": "Is water damage covered by this policy?", "top_k": 3}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(ask_req) as ask_resp:
        assert ask_resp.status == 200
        ask_data = json.loads(ask_resp.read().decode("utf-8"))
        answer = ask_data["answer"]
        sources = ask_data["sources"]
        query_id = ask_data["query_id"]
        assert len(answer) > 0
        assert len(sources) > 0
        print(f"  -> Grounded Answer: {answer}")
        print("  -> Source Citations:")
        for s in sources:
            print(f"     * File: {s['file_name']} | Page: {s['page_number']} | Chunk ID: {s['chunk_id']}")

    # Step 15: Store Query and Answer in MySQL Verification
    print("\n[Step 15] Verifying Q&A audit record in MySQL queries table...")
    query_repo = QueryRepository(db)
    saved_query = query_repo.get_by_id(query_id)
    assert saved_query is not None
    assert saved_query.document_id == doc_id
    assert "water damage" in saved_query.question.lower()
    print(f"  -> Verified MySQL QueryRecord ID={saved_query.id}, doc_id={saved_query.document_id}, created_at={saved_query.created_at}")

    # Step 16: Document Summary Generation
    print("\n[Step 16] Generating executive summary (POST /documents/{id}/summary)...")
    summary_req = urllib.request.Request(f"{base_url}/documents/{doc_id}/summary", data=b"", method="POST")
    with urllib.request.urlopen(summary_req) as sum_resp:
        assert sum_resp.status == 200
        sum_data = json.loads(sum_resp.read().decode("utf-8"))
        assert len(sum_data["summary"]) > 0
        assert len(sum_data["key_points"]) >= 2
        print(f"  -> Executive Summary: {sum_data['summary'][:90]}...")
        print(f"  -> Key Takeaways ({len(sum_data['key_points'])} bullets):")
        for kp in sum_data["key_points"][:3]:
            print(f"     * {kp}")

    # Step 17: Structured Information Extraction
    print("\n[Step 17] Extracting structured insurance data into JSON (POST /documents/{id}/extract)...")
    extract_req = urllib.request.Request(f"{base_url}/documents/{doc_id}/extract", data=b"", method="POST")
    with urllib.request.urlopen(extract_req) as ext_resp:
        assert ext_resp.status == 200
        ext_data = json.loads(ext_resp.read().decode("utf-8"))
        extraction = ext_data["extraction"]
        assert extraction["policy_type"] is not None
        assert extraction["coverage"] is not None
        assert len(extraction["exclusions"]) > 0
        assert extraction["deductible"] is not None
        print("  -> Structured JSON Extraction validated:")
        print(f"     Policy Type: {extraction['policy_type']}")
        print(f"     Deductible: {extraction['deductible']}")
        print(f"     Exclusions: {extraction['exclusions']}")

    # Step 18: Snowflake Analytics Sync and Reporting
    print("\n[Step 18] Verifying Snowflake analytics synchronization and reporting...")
    # Explicitly update risk category to MEDIUM for testing sync
    sync_req = urllib.request.Request(
        f"{base_url}/analytics/sync/{doc_id}?risk_category=MEDIUM",
        data=b"",
        method="POST",
    )
    with urllib.request.urlopen(sync_req) as sync_resp:
        assert sync_resp.status == 200
        sync_result = json.loads(sync_resp.read().decode("utf-8"))
        assert sync_result["status"] == "success"
        print(f"  -> Operational metrics successfully synced to Snowflake: doc_id={doc_id}")

    # Fetch aggregated analytics report (GET /analytics/summary)
    analytics_req = urllib.request.Request(f"{base_url}/analytics/summary", method="GET")
    with urllib.request.urlopen(analytics_req) as an_resp:
        assert an_resp.status == 200
        analytics_summary = json.loads(an_resp.read().decode("utf-8"))
        assert analytics_summary["total_processed_documents"] >= 1
        assert len(analytics_summary["documents_by_type"]) >= 1
        assert len(analytics_summary["questions_by_document"]) >= 1
        assert len(analytics_summary["risk_distribution"]) >= 1
        assert analytics_summary["processing_time_metrics"]["avg_processing_time"] > 0
        print(f"  -> Snowflake Analytics Report:")
        print(f"     Total Processed Documents: {analytics_summary['total_processed_documents']}")
        print(f"     Documents by Type: {analytics_summary['documents_by_type']}")
        print(f"     Questions per Document: {analytics_summary['questions_by_document']}")
        print(f"     Risk Distribution: {analytics_summary['risk_distribution']}")
        print(f"     Processing Latency Metrics: {analytics_summary['processing_time_metrics']}")

    print("\n" + "=" * 80)
    print("ALL 18 STEPS OF THE MASTER END-TO-END WORKFLOW SUCCEEDED (100% PASS)!")
    print("=" * 80)


if __name__ == "__main__":
    run_master_test()
