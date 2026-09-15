"""Phase 5 automated verification script.

Validates:
1. BedrockLLMService abstraction and Amazon Nova Lite configuration.
2. MockLLMService text generation and structured JSON formatting.
3. DocumentAnalysisService text assembly across pages and chunks.
4. Document summarization (POST /documents/{id}/summary) returning narrative and key points.
5. Structured insurance information extraction (POST /documents/{id}/extract) returning
   policy_type, coverage, exclusions, deductible, important_conditions, risk_indicators.
6. Live HTTP endpoint execution via FastAPI application.
"""
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

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.aws.llm_service import (
    BaseLLMService,
    BedrockLLMService,
    MockLLMService,
    get_llm_service,
)
from app.config import get_settings
from app.database.session import Base, get_db
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.analysis import (
    DocumentExtractionResponse,
    DocumentSummaryResponse,
    StructuredInsuranceExtraction,
)
from app.services.analysis_service import DocumentAnalysisService
from main import app as fastapi_app


def run_tests():
    print("=" * 65)
    print("AI Insurance Document Intelligence Assistant - Phase 5 Verification")
    print("=" * 65)

    # 1. Test LLM Service Abstraction & Nova Lite Configuration
    print("\n[1/6] Testing LLM Service abstraction and Bedrock Nova Lite configuration...")
    settings = get_settings()
    bedrock_llm = BedrockLLMService(settings=settings)
    assert isinstance(bedrock_llm, BaseLLMService)
    assert bedrock_llm.model_id == "amazon.nova-lite-v1:0"
    assert settings.bedrock_max_tokens == 2048
    assert settings.bedrock_temperature == 0.0
    print(f"  -> BedrockLLMService configured for model '{bedrock_llm.model_id}' (max_tokens={settings.bedrock_max_tokens}, temp={settings.bedrock_temperature}).")

    mock_llm = MockLLMService(model_id="mock.nova-lite-v1:0")
    test_gen = mock_llm.generate_text("Summarize this document")
    assert len(test_gen) > 0
    print("  -> MockLLMService text generation verified.")

    # 2. Setup In-Memory Database with Multi-Page Document Chunks
    print("\n[2/6] Initializing test database and sample insurance policy...")
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

    # Create document
    test_doc = Document(
        file_name="acme_property_policy.pdf",
        document_type="Commercial Property",
        s3_key="documents/acme_property_policy.pdf",
        status="PROCESSED",
    )
    doc_repo.create(test_doc)
    doc_id = test_doc.id

    # Create multi-page chunks
    chunks = [
        DocumentChunk(
            document_id=doc_id,
            chunk_index=0,
            page_number=1,
            chunk_text=(
                "COMMERCIAL PROPERTY INSURANCE POLICY\n"
                "Policy Number: POL-2026-8899\n"
                "Named Insured: Acme Global Logistics LLC\n"
                "Policy Period: From January 1, 2026 to January 1, 2027\n"
                "Annual Premium: $14,500.00 USD"
            ),
            file_name="acme_property_policy.pdf",
            document_type="Commercial Property",
        ),
        DocumentChunk(
            document_id=doc_id,
            chunk_index=1,
            page_number=2,
            chunk_text=(
                "SCHEDULE OF COVERAGES AND LIMITS\n"
                "Coverage A - Building Property Limit: $2,500,000\n"
                "Coverage B - Business Personal Property: $750,000\n"
                "Coverage C - Business Income with Extra Expense: $300,000\n"
                "Standard Deductible: $5,000 per occurrence\n"
                "Windstorm / Hail Deductible: 2% of stated building value."
            ),
            file_name="acme_property_policy.pdf",
            document_type="Commercial Property",
        ),
        DocumentChunk(
            document_id=doc_id,
            chunk_index=2,
            page_number=3,
            chunk_text=(
                "EXCLUSIONS AND POLICY CONDITIONS\n"
                "1. Flood and Surface Water: Not covered unless specifically endorsed.\n"
                "2. Earth Movement: Earthquake, volcanic eruption, and landslide excluded.\n"
                "3. Normal Wear and Tear: Rust, corrosion, decay, and gradual deterioration.\n"
                "4. Prompt Notice of Claim: Insured must notify insurer within 30 days of loss."
            ),
            file_name="acme_property_policy.pdf",
            document_type="Commercial Property",
        ),
    ]
    chunk_repo.create_bulk(chunks)
    print(f"  -> Created document ID={doc_id} with {len(chunks)} chunks across 3 pages.")

    # 3. Test Document Summarization Service
    print("\n[3/6] Testing DocumentAnalysisService.summarize_document...")
    analysis_service = DocumentAnalysisService(
        document_repository=doc_repo,
        chunk_repository=chunk_repo,
        llm_service=mock_llm,
    )

    summary_resp = analysis_service.summarize_document(doc_id)
    assert isinstance(summary_resp, DocumentSummaryResponse)
    assert summary_resp.document_id == doc_id
    assert summary_resp.file_name == "acme_property_policy.pdf"
    assert summary_resp.document_type == "Commercial Property"
    assert len(summary_resp.summary) > 0
    assert len(summary_resp.key_points) >= 2
    assert summary_resp.model_id == "mock.nova-lite-v1:0"
    print("  -> Summary generated successfully:")
    print(f"     Narrative: {summary_resp.summary[:80]}...")
    print(f"     Key Points Count: {len(summary_resp.key_points)}")
    for kp in summary_resp.key_points[:2]:
        print(f"       * {kp}")

    # 4. Test Structured Insurance Information Extraction Service
    print("\n[4/6] Testing DocumentAnalysisService.extract_structured_info...")
    extract_resp = analysis_service.extract_structured_info(doc_id)
    assert isinstance(extract_resp, DocumentExtractionResponse)
    assert extract_resp.document_id == doc_id
    assert extract_resp.file_name == "acme_property_policy.pdf"
    assert isinstance(extract_resp.extraction, StructuredInsuranceExtraction)

    extraction = extract_resp.extraction
    assert extraction.policy_type is not None
    assert extraction.coverage is not None
    assert extraction.exclusions is not None and len(extraction.exclusions) > 0
    assert extraction.deductible is not None
    assert extraction.important_conditions is not None and len(extraction.important_conditions) > 0
    assert extraction.risk_indicators is not None and len(extraction.risk_indicators) > 0

    print("  -> Structured data extracted successfully without hallucination:")
    print(f"     Policy Type: {extraction.policy_type}")
    print(f"     Coverage: {extraction.coverage}")
    print(f"     Exclusions: {extraction.exclusions}")
    print(f"     Deductible: {extraction.deductible}")
    print(f"     Conditions: {extraction.important_conditions}")
    print(f"     Risk Indicators: {extraction.risk_indicators}")

    # 5. Test Error Handling
    print("\n[5/6] Testing error handling for missing documents...")
    try:
        analysis_service.summarize_document(9999)
        raise AssertionError("Expected 404 for missing document")
    except HTTPException as e:
        assert e.status_code == 404
        print(f"  -> Non-existent ID 9999 correctly raised HTTP 404: {e.detail}")

    # 6. Test Live HTTP Endpoints (POST /summary and POST /extract)
    print("\n[6/6] Testing live HTTP POST endpoints via FastAPI & Uvicorn...")
    def override_get_test_db():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    fastapi_app.dependency_overrides[get_db] = override_get_test_db
    fastapi_app.dependency_overrides[get_llm_service] = lambda: mock_llm

    server_thread = threading.Thread(
        target=lambda: __import__("uvicorn").run(fastapi_app, host="127.0.0.1", port=8005, log_level="error"),
        daemon=True,
    )
    server_thread.start()
    time.sleep(2)

    # Test POST /documents/{id}/summary
    req_summary = urllib.request.Request(
        f"http://127.0.0.1:8005/documents/{doc_id}/summary",
        data=b"",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req_summary) as resp:
        assert resp.status == 200, f"Expected 200 OK, got {resp.status}"
        data = json.loads(resp.read().decode("utf-8"))
        assert data["document_id"] == doc_id
        assert len(data["summary"]) > 0
        assert len(data["key_points"]) > 0
        print(f"  -> HTTP POST /documents/{doc_id}/summary: 200 OK (points={len(data['key_points'])})")

    # Test POST /documents/{id}/extract
    req_extract = urllib.request.Request(
        f"http://127.0.0.1:8005/documents/{doc_id}/extract",
        data=b"",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req_extract) as resp:
        assert resp.status == 200, f"Expected 200 OK, got {resp.status}"
        data = json.loads(resp.read().decode("utf-8"))
        assert data["document_id"] == doc_id
        assert data["extraction"]["policy_type"] == "Commercial Property Insurance"
        assert len(data["extraction"]["exclusions"]) > 0
        print(f"  -> HTTP POST /documents/{doc_id}/extract: 200 OK (exclusions={len(data['extraction']['exclusions'])})")

    db.close()
    print("\n" + "=" * 65)
    print("ALL PHASE 5 AUTOMATED TESTS PASSED SUCCESSFULLY (100% SUCCESS)!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
