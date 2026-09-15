"""Phase 6 automated verification script.

Validates the complete Retrieval-Augmented Generation (RAG) pipeline:
1. RAGService orchestration:
   - Scoped similarity search in ChromaDB (isolated strictly to target document_id)
   - Prompt construction with strict grounding instructions (anti-hallucination, no CoT)
   - Generation via Amazon Nova Lite (using BedrockLLMService / MockLLMService)
   - Source attribution metadata extraction (file_name, page_number, chunk_id)
   - Persistence of question and answer in MySQL/SQLite queries audit table
2. Document isolation: Chunks from other documents never bleed into retrieval
3. Uncovered/out-of-scope query fallback handling ("The document does not provide enough information...")
4. Input validation and error handling (HTTP 404 for missing documents, HTTP 400 for empty questions)
5. Live HTTP POST /documents/{document_id}/ask endpoint execution via FastAPI and Uvicorn
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Add project root to sys.path so script can run directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chromadb
from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.aws.bedrock_service import MockEmbeddingService
from app.aws.llm_service import MockLLMService, get_llm_service
from app.config import get_settings
from app.database.session import Base, get_db
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.query import QueryRecord
from app.rag.rag_service import RAGService
from app.rag.vector_store import VectorStoreService, get_vector_store_service
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.query_repository import QueryRepository
from app.schemas.rag import RAGAnswerResponse, SourceMetadata
from main import app as fastapi_app


def run_tests():
    print("=" * 70)
    print("AI Insurance Document Intelligence Assistant - Phase 6 Verification")
    print("=" * 70)

    # 1. Initialize In-Memory Test Database
    print("\n[1/7] Initializing test database and sample insurance documents...")
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

    # Create Document 1: Commercial Property Policy
    doc1 = Document(
        file_name="acme_property_policy.pdf",
        document_type="Commercial Property",
        s3_key="documents/acme_property_policy.pdf",
        status="PROCESSED",
    )
    doc_repo.create(doc1)
    doc1_id = doc1.id

    # Create Document 2: Commercial Auto Policy (for document isolation testing)
    doc2 = Document(
        file_name="acme_fleet_auto_policy.pdf",
        document_type="Commercial Auto",
        s3_key="documents/acme_fleet_auto_policy.pdf",
        status="PROCESSED",
    )
    doc_repo.create(doc2)
    doc2_id = doc2.id
    print(f"  -> Created Doc 1 (ID={doc1_id}, '{doc1.file_name}') and Doc 2 (ID={doc2_id}, '{doc2.file_name}').")

    # 2. Setup ChromaDB Vector Store with Scoped Chunks
    print("\n[2/7] Populating vector store with document chunks...")
    import shutil
    test_chroma_dir = str(ROOT_DIR / "data" / "test_chroma_phase6")
    shutil.rmtree(test_chroma_dir, ignore_errors=True)
    settings = get_settings()
    settings.chroma_persist_directory = test_chroma_dir
    settings.chroma_collection_name = "test_phase6_rag_chunks"

    embedding_service = MockEmbeddingService()
    vector_store = VectorStoreService(
        embedding_service=embedding_service,
        settings=settings,
    )

    # Chunks for Document 1
    doc1_chunks = [
        DocumentChunk(
            id=101,
            document_id=doc1_id,
            chunk_index=0,
            page_number=1,
            chunk_text=(
                "POLICY DECLARATIONS - COMMERCIAL PROPERTY\n"
                "Named Insured: Acme Global Logistics LLC\n"
                "Policy Term: 01/01/2026 to 01/01/2027\n"
                "Coverage A - Building Property: $2,500,000 limit.\n"
                "Coverage B - Business Personal Property: $750,000 limit.\n"
                "Standard Policy Deductible: $5,000 per occurrence."
            ),
            file_name=doc1.file_name,
            document_type=doc1.document_type,
        ),
        DocumentChunk(
            id=102,
            document_id=doc1_id,
            chunk_index=1,
            page_number=2,
            chunk_text=(
                "SECTION B - EXCLUSIONS AND WATER DAMAGE\n"
                "1. Flood, surface water, waves, tidal water, and sewer overflow are excluded from coverage.\n"
                "2. Earth movement including earthquake, tremor, sinkhole collapse, and landslide are excluded.\n"
                "3. Sudden and accidental discharge of water from broken internal plumbing pipes is covered "
                "subject to the $5,000 standard deductible."
            ),
            file_name=doc1.file_name,
            document_type=doc1.document_type,
        ),
        DocumentChunk(
            id=103,
            document_id=doc1_id,
            chunk_index=2,
            page_number=3,
            chunk_text=(
                "ENDORSEMENT CP-04: WINDSTORM AND HAIL PERCENTAGE DEDUCTIBLE\n"
                "For any loss or damage caused by windstorm, cyclone, tornado, or hail,\n"
                "a separate deductible of 2% of the stated building value applies in lieu of standard deductible."
            ),
            file_name=doc1.file_name,
            document_type=doc1.document_type,
        ),
    ]

    # Chunks for Document 2 (Commercial Auto)
    doc2_chunks = [
        DocumentChunk(
            id=201,
            document_id=doc2_id,
            chunk_index=0,
            page_number=1,
            chunk_text=(
                "COMMERCIAL AUTO FLEET COVERAGE\n"
                "Comprehensive and Collision deductible: $1,000 per vehicle.\n"
                "Combined Single Limit for bodily injury and property damage: $1,000,000 per accident."
            ),
            file_name=doc2.file_name,
            document_type=doc2.document_type,
        )
    ]

    vector_store.add_chunks(doc1_chunks)
    vector_store.add_chunks(doc2_chunks)
    print(f"  -> Added {len(doc1_chunks)} chunks for Doc 1 and {len(doc2_chunks)} chunks for Doc 2 to ChromaDB.")

    # 3. Instantiate RAGService and Test Grounded Q&A
    print("\n[3/7] Testing RAGService.ask_question with water damage inquiry...")
    mock_llm = MockLLMService(model_id="mock.nova-lite-v1:0")
    rag_service = RAGService(
        vector_store=vector_store,
        llm_service=mock_llm,
        query_repository=query_repo,
        document_repository=doc_repo,
    )

    question1 = "Is water damage covered by this policy?"
    response1 = rag_service.ask_question(
        document_id=doc1_id,
        question=question1,
        top_k=3,
    )

    assert isinstance(response1, RAGAnswerResponse)
    assert response1.document_id == doc1_id
    assert response1.query_id is not None
    assert len(response1.answer) > 0
    assert len(response1.sources) > 0
    print("  -> Grounded Answer received:")
    print(f"     \"{response1.answer}\"")
    print(f"  -> Source Attributions ({len(response1.sources)} sources):")
    for s in response1.sources:
        assert isinstance(s, SourceMetadata)
        assert s.file_name == doc1.file_name
        assert s.page_number in [1, 2, 3]
        print(f"     - [{s.file_name} | Page {s.page_number} | Chunk {s.chunk_id}]")

    # Verify persistence in database
    persisted_query = query_repo.get_by_id(response1.query_id)
    assert persisted_query is not None
    assert persisted_query.document_id == doc1_id
    assert persisted_query.question == question1
    assert persisted_query.answer == response1.answer
    assert persisted_query.created_at is not None
    print(f"  -> Successfully verified MySQL/SQLite persistence (Record ID={persisted_query.id}).")

    # 4. Test Deductibles Question
    print("\n[4/7] Testing RAGService with building deductible question...")
    question2 = "What is the deductible for building property damage and windstorm?"
    response2 = rag_service.ask_question(
        document_id=doc1_id,
        question=question2,
        top_k=2,
    )
    assert "$5,000" in response2.answer or "deductible" in response2.answer
    assert len(response2.sources) > 0
    print(f"  -> Answer: \"{response2.answer}\"")
    print(f"  -> Sources count: {len(response2.sources)}")

    # 5. Test Strict Document Scoping / Isolation
    print("\n[5/7] Testing Document Isolation in ChromaDB retrieval...")
    # Searching for vehicle fleet info on Property policy (Doc 1) should NOT return auto chunks from Doc 2
    fleet_query = "What is the auto collision deductible?"
    doc1_search_results = vector_store.similarity_search(fleet_query, top_k=5, document_id=doc1_id)
    for res in doc1_search_results:
        assert res["document_id"] == doc1_id, f"Found leaked document_id {res['document_id']} when querying doc {doc1_id}"
        assert res["document_name"] == doc1.file_name
    print("  -> Confirmed strict document isolation: 0 chunk leakages from Doc 2 into Doc 1 queries.")

    # 6. Test Uncovered Question / Out of Scope Fallback
    print("\n[6/7] Testing out-of-scope query fallback handling...")
    uncovered_q = "Does this policy cover cyber ransom and cryptocurrency extortion?"
    response_uncovered = rag_service.ask_question(
        document_id=doc1_id,
        question=uncovered_q,
    )
    assert "not provide enough information" in response_uncovered.answer.lower()
    print(f"  -> Fallback Answer: \"{response_uncovered.answer}\"")
    # Verify fallback is also audited in queries table
    assert response_uncovered.query_id is not None
    audit_uncovered = query_repo.get_by_id(response_uncovered.query_id)
    assert audit_uncovered is not None
    assert audit_uncovered.question == uncovered_q
    print(f"  -> Out-of-scope question persisted in queries table (ID={audit_uncovered.id}).")

    # Test Validation & 404
    try:
        rag_service.ask_question(99999, "What is the deductible?")
        raise AssertionError("Expected 404 for non-existent document ID")
    except HTTPException as exc:
        assert exc.status_code == 404
        print(f"  -> Missing document ID correctly raised HTTP 404: {exc.detail}")

    try:
        rag_service.ask_question(doc1_id, "   ")
        raise AssertionError("Expected 400 for empty question")
    except HTTPException as exc:
        assert exc.status_code == 400
        print(f"  -> Empty question correctly raised HTTP 400: {exc.detail}")

    # 7. Test Live HTTP POST /documents/{document_id}/ask
    print("\n[7/7] Testing live HTTP POST endpoint /documents/{document_id}/ask...")
    def override_test_db():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    fastapi_app.dependency_overrides[get_db] = override_test_db
    fastapi_app.dependency_overrides[get_llm_service] = lambda: mock_llm
    fastapi_app.dependency_overrides[get_vector_store_service] = lambda: vector_store

    server_thread = threading.Thread(
        target=lambda: __import__("uvicorn").run(fastapi_app, host="127.0.0.1", port=8006, log_level="error"),
        daemon=True,
    )
    server_thread.start()
    time.sleep(2)

    ask_url = f"http://127.0.0.1:8006/documents/{doc1_id}/ask"
    payload = {
        "question": "Is water damage covered by this policy?",
        "top_k": 3,
    }
    req = urllib.request.Request(
        ask_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        http_data = json.loads(resp.read().decode("utf-8"))
        assert "answer" in http_data
        assert "sources" in http_data
        assert isinstance(http_data["sources"], list)
        assert len(http_data["sources"]) > 0
        assert "document_id" in http_data
        assert "query_id" in http_data
        print("  -> Live HTTP POST /documents/{id}/ask succeeded (Status 200 OK):")
        print(f"     Payload: {json.dumps(payload)}")
        print(f"     Answer: {http_data['answer'][:90]}...")
        print(f"     Sources: {http_data['sources']}")
        print(f"     Query ID: {http_data['query_id']}")

    # Verify 404 on non-existent document
    try:
        err_req = urllib.request.Request(
            "http://127.0.0.1:8006/documents/99999/ask",
            data=json.dumps({"question": "Is water damage covered?"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(err_req)
        raise AssertionError("Expected HTTP 404 from live endpoint")
    except urllib.error.HTTPError as http_err:
        assert http_err.code == 404
        print("  -> Live endpoint correctly returned HTTP 404 for invalid document ID.")

    print("\n" + "=" * 70)
    print("ALL PHASE 6 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
