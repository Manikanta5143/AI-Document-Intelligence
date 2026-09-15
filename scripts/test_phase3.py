"""Phase 3 automated verification script.

Validates:
1. PyMuPDF PDF text extraction and cleaning page-by-page.
2. Text chunker with configurable chunk size and overlap.
3. Database persistence of chunks preserving document_id, chunk_id, page_number, text, file_name, document_type.
4. DocumentService end-to-end processing pipeline and file validation.
"""
import io
import sys
from pathlib import Path

# Add project root to sys.path so script can be run directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pymupdf
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.session import Base
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.document_service import DocumentService
from app.services.pdf_extractor import PDFExtractor
from app.services.text_chunker import TextChunker


def create_synthetic_insurance_pdf() -> bytes:
    """Create a 3-page synthetic insurance policy PDF in memory for testing."""
    doc = pymupdf.open()

    # Page 1: Policy Declarations
    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "COMMERCIAL PROPERTY INSURANCE POLICY\n"
        "Policy Number: POL-2026-8899\n"
        "Named Insured: Acme Global Logistics LLC\n"
        "Policy Period: From January 1, 2026 to January 1, 2027\n"
        "Annual Premium: $14,500.00 USD\n\n"
        "This policy provides coverage against direct physical loss or damage "
        "to covered property at described premises, subject to all terms and conditions.",
        fontsize=11,
    )

    # Page 2: Coverage Limits
    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        "SCHEDULE OF COVERAGES AND LIMITS\n"
        "Coverage A - Building Property Limit: $2,500,000\n"
        "Coverage B - Business Personal Property: $750,000\n"
        "Coverage C - Business Income with Extra Expense: $300,000\n"
        "Standard Deductible: $5,000 per occurrence\n"
        "Windstorm / Hail Deductible: 2% of stated building value.",
        fontsize=11,
    )

    # Page 3: Exclusions
    p3 = doc.new_page()
    p3.insert_text(
        (50, 72),
        "EXCLUSIONS AND POLICY CONDITIONS\n"
        "1. Flood and Surface Water: Not covered unless specifically endorsed.\n"
        "2. Earth Movement: Earthquake, volcanic eruption, and landslide excluded.\n"
        "3. Normal Wear and Tear: Rust, corrosion, decay, and gradual deterioration.\n"
        "4. Prompt Notice of Claim: Insured must notify insurer within 30 days of loss.",
        fontsize=11,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class StubS3Service:
    """Stub S3Service for isolated testing without network latency or external AWS calls."""

    def __init__(self):
        self.uploaded_keys = []

    def generate_s3_key(self, file_name: str) -> str:
        return f"documents/test_stub_{file_name}"

    def upload_fileobj(self, file_obj, s3_key: str, content_type: str = "application/pdf") -> str:
        data = file_obj.read()
        self.uploaded_keys.append({"key": s3_key, "size": len(data), "content_type": content_type})
        return s3_key


def run_tests():
    print("=" * 65)
    print("AI Insurance Document Intelligence Assistant - Phase 3 Verification")
    print("=" * 65)

    # 1. Generate Synthetic Insurance PDF
    print("\n[1/6] Creating synthetic 3-page insurance policy PDF in-memory...")
    pdf_bytes = create_synthetic_insurance_pdf()
    assert pdf_bytes.startswith(b"%PDF-"), "Invalid PDF generated"
    print(f"  -> Generated valid PDF ({len(pdf_bytes)} bytes)")

    # 2. Test PyMuPDF Text Extraction & Cleaning
    print("\n[2/6] Testing PyMuPDF text extraction & sanitization (PDFExtractor)...")
    extractor = PDFExtractor()
    pages_data = extractor.extract_pages(pdf_bytes)
    assert len(pages_data) == 3, f"Expected 3 pages, got {len(pages_data)}"
    assert pages_data[0]["page_number"] == 1
    assert "COMMERCIAL PROPERTY INSURANCE POLICY" in pages_data[0]["text"]
    assert pages_data[1]["page_number"] == 2
    assert "Coverage A - Building Property Limit: $2,500,000" in pages_data[1]["text"]
    assert pages_data[2]["page_number"] == 3
    assert "EXCLUSIONS AND POLICY CONDITIONS" in pages_data[2]["text"]
    print("  -> Page 1, 2, and 3 extracted and cleaned with exact text matching.")

    # 3. Test TextChunker with Configurable Size & Overlap
    print("\n[3/6] Testing text chunker with overlap (TextChunker)...")
    chunker = TextChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_document_pages(pages_data)
    assert len(chunks) > 0, "No chunks generated"
    for c in chunks:
        assert "chunk_index" in c
        assert "page_number" in c
        assert "chunk_text" in c
        assert len(c["chunk_text"]) > 0
    print(f"  -> Generated {len(chunks)} chunks across 3 pages (chunk_size=150, overlap=30).")

    # 4. Test Database Persistence & Chunk Model
    print("\n[4/6] Testing ChunkRepository & DocumentChunk database persistence...")
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)

    test_doc = Document(
        file_name="acme_logistics_policy.pdf",
        document_type="Commercial Property",
        s3_key="documents/acme_logistics_policy.pdf",
        status="PROCESSING",
    )
    doc_repo.create(test_doc)
    assert test_doc.id == 1

    chunk_models = [
        DocumentChunk(
            document_id=test_doc.id,
            chunk_index=c["chunk_index"],
            page_number=c["page_number"],
            chunk_text=c["chunk_text"],
            file_name=test_doc.file_name,
            document_type=test_doc.document_type,
        )
        for c in chunks
    ]
    saved_chunks = chunk_repo.create_bulk(chunk_models)
    assert len(saved_chunks) == len(chunks)

    # Verify all required chunk fields are preserved
    sample_chunk = saved_chunks[0]
    assert sample_chunk.document_id == 1
    assert sample_chunk.id is not None
    assert sample_chunk.chunk_index == 0
    assert sample_chunk.page_number == 1
    assert sample_chunk.chunk_text != ""
    assert sample_chunk.file_name == "acme_logistics_policy.pdf"
    assert sample_chunk.document_type == "Commercial Property"

    retrieved = chunk_repo.get_by_document_id(test_doc.id)
    assert len(retrieved) == len(chunks)
    assert chunk_repo.count_by_document_id(test_doc.id) == len(chunks)
    print(f"  -> Successfully verified all {len(retrieved)} chunk records in DB.")

    # 5. Test Full Document Ingestion Pipeline via DocumentService
    print("\n[5/6] Testing DocumentService.process_and_upload_document end-to-end...")
    stub_s3 = StubS3Service()
    service = DocumentService(
        document_repository=doc_repo,
        chunk_repository=chunk_repo,
        s3_service=stub_s3,
        pdf_extractor=extractor,
        text_chunker=chunker,
    )

    result = service.process_and_upload_document(
        file_name="fleet_auto_policy.pdf",
        file_bytes=pdf_bytes,
        document_type="Auto Policy",
    )
    assert result.document_id == 2
    assert result.status == "PROCESSED"
    assert result.total_pages == 3
    assert result.total_chunks == len(chunks)
    assert result.s3_key.startswith("documents/test_stub_fleet_auto_policy.pdf")
    assert len(result.chunks) == len(chunks)
    assert len(stub_s3.uploaded_keys) == 1
    print(f"  -> Pipeline executed: doc_id={result.document_id}, status={result.status}, s3_key={result.s3_key}")

    # 6. Test File Validation Errors
    print("\n[6/6] Testing validation and error handling...")
    # Non-PDF extension
    try:
        service.process_and_upload_document("notes.txt", b"some text", "Policy")
        raise AssertionError("Expected 400 for non-pdf")
    except HTTPException as e:
        assert e.status_code == 400
        print(f"  -> Blocked non-PDF file: {e.detail}")

    # Empty file
    try:
        service.process_and_upload_document("empty.pdf", b"", "Policy")
        raise AssertionError("Expected 400 for empty file")
    except HTTPException as e:
        assert e.status_code == 400
        print(f"  -> Blocked empty PDF file: {e.detail}")

    # Corrupt PDF header
    try:
        service.process_and_upload_document("corrupt.pdf", b"NOT_A_PDF_STREAM", "Policy")
        raise AssertionError("Expected 400 for corrupt PDF")
    except HTTPException as e:
        assert e.status_code == 400
        print(f"  -> Blocked corrupt PDF header: {e.detail}")

    # 7. Test HTTP Multipart Upload via FastAPI Endpoint
    print("\n[7/7] Testing HTTP POST /documents/upload multipart endpoint...")
    import json
    import threading
    import time
    import urllib.request
    from sqlalchemy.pool import StaticPool
    import uvicorn
    from app.database.session import get_db
    from main import app as fastapi_app
    import app.services.document_service as doc_service_module

    http_engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=http_engine)
    HttpSession = sessionmaker(bind=http_engine)

    def override_get_http_db():
        session = HttpSession()
        try:
            yield session
        finally:
            session.close()

    fastapi_app.dependency_overrides[get_db] = override_get_http_db
    doc_service_module.S3Service = StubS3Service

    server_thread = threading.Thread(
        target=lambda: uvicorn.run(fastapi_app, host="127.0.0.1", port=8003, log_level="error"),
        daemon=True,
    )
    server_thread.start()
    time.sleep(2)

    boundary = "----WebKitFormBoundaryPhase3Test"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="file"; filename="workers_comp_policy.pdf"\r\n')
    body.extend(b"Content-Type: application/pdf\r\n\r\n")
    body.extend(pdf_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="document_type"\r\n\r\n')
    body.extend(b"Workers Compensation\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        "http://127.0.0.1:8003/documents/upload",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 201, f"Expected 201 Created, got {resp.status}"
        resp_data = json.loads(resp.read().decode())
        assert resp_data["status"] == "PROCESSED"
        assert resp_data["total_pages"] == 3
        assert resp_data["total_chunks"] == 3  # 1 chunk per page with default chunk_size=1000
        assert resp_data["file_name"] == "workers_comp_policy.pdf"
        assert resp_data["document_type"] == "Workers Compensation"
        assert len(resp_data["chunks"]) == 3
        print(f"  -> HTTP 201 Created: doc_id={resp_data['document_id']}, chunks={resp_data['total_chunks']}")

    db.close()
    print("\n" + "=" * 65)
    print("ALL PHASE 3 AUTOMATED TESTS PASSED SUCCESSFULLY (100% SUCCESS)!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
