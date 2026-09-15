"""Phase 4 automated verification script.

Validates:
1. EmbeddingService abstraction (BaseEmbeddingService, MockEmbeddingService, BedrockEmbeddingService).
2. ChromaDB persistent vector database setup and collection management.
3. Chunk indexing with required metadata (embedding, chunk text, document ID, document name, page number, chunk ID, document type).
4. Semantic similarity search returning top-5 relevant chunks with cosine similarity score.
5. Metadata filtering by document_id and document_type.
6. Live HTTP POST /search endpoint execution via FastAPI application.
"""
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Add project root to sys.path so script can be run directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.aws.bedrock_service import (
    BaseEmbeddingService,
    BedrockEmbeddingService,
    MockEmbeddingService,
)
from app.config import get_settings
from app.models.chunk import DocumentChunk
from app.rag.vector_store import VectorStoreService, get_vector_store_service
from main import app as fastapi_app

TEST_CHROMA_DIR = "./data/test_chromadb"


def cleanup_test_dir():
    if os.path.exists(TEST_CHROMA_DIR):
        try:
            shutil.rmtree(TEST_CHROMA_DIR)
        except Exception:
            pass


def run_tests():
    print("=" * 65)
    print("AI Insurance Document Intelligence Assistant - Phase 4 Verification")
    print("=" * 65)

    cleanup_test_dir()

    # 1. Test Embedding Service Abstraction
    print("\n[1/6] Testing EmbeddingService abstraction and MockEmbeddingService...")
    embedding_service = MockEmbeddingService(dimension=1024)
    assert isinstance(embedding_service, BaseEmbeddingService)

    sample_query = "What is the deductible for building property damage?"
    query_vector = embedding_service.embed_text(sample_query)
    assert len(query_vector) == 1024, f"Expected dimension 1024, got {len(query_vector)}"
    assert any(x != 0 for x in query_vector), "Embedding vector is all zeroes"

    doc_texts = [
        "Building Property Coverage Limit is $2,500,000 with a $5,000 deductible.",
        "Flood and Earth movement are strictly excluded under standard commercial terms.",
        "Business Personal Property is covered up to $750,000 at described premises.",
    ]
    doc_vectors = embedding_service.embed_documents(doc_texts)
    assert len(doc_vectors) == 3
    assert len(doc_vectors[0]) == 1024
    print(f"  -> Generated query embedding (dim={len(query_vector)}) and {len(doc_vectors)} doc embeddings.")

    # Test BedrockEmbeddingService constructor & configuration
    settings = get_settings()
    bedrock_service = BedrockEmbeddingService(settings=settings)
    assert bedrock_service.model_id == settings.bedrock_embedding_model_id
    payload = bedrock_service._build_payload("test prompt")
    assert payload["dimensions"] == 1024
    assert payload["normalize"] is True
    print(f"  -> BedrockEmbeddingService configured for model '{bedrock_service.model_id}'.")

    # 2. Test ChromaDB Persistence & Collection Initialization
    print("\n[2/6] Testing ChromaDB local persistence in '%s'..." % TEST_CHROMA_DIR)
    settings.chroma_persist_directory = TEST_CHROMA_DIR
    settings.chroma_collection_name = "test_insurance_chunks"

    vector_store = VectorStoreService(
        embedding_service=embedding_service,
        settings=settings,
    )
    assert os.path.exists(TEST_CHROMA_DIR), "ChromaDB directory was not created"
    print(f"  -> ChromaDB client initialized and persisted at '{TEST_CHROMA_DIR}'.")

    # 3. Test Ingesting Chunks with All Required Metadata Fields
    print("\n[3/6] Testing indexing chunks with metadata into ChromaDB...")
    test_chunks = [
        DocumentChunk(
            id=101,
            document_id=1,
            chunk_index=0,
            page_number=1,
            chunk_text="Commercial Property Policy Declarations: Named Insured Acme Global Logistics.",
            file_name="acme_property_policy.pdf",
            document_type="Commercial Property",
        ),
        DocumentChunk(
            id=102,
            document_id=1,
            chunk_index=1,
            page_number=2,
            chunk_text="Coverage A - Building Property Limit: $2,500,000. Standard Deductible: $5,000 per occurrence.",
            file_name="acme_property_policy.pdf",
            document_type="Commercial Property",
        ),
        DocumentChunk(
            id=103,
            document_id=1,
            chunk_index=2,
            page_number=3,
            chunk_text="Exclusions: Flood and surface water damage are not covered under this property form.",
            file_name="acme_property_policy.pdf",
            document_type="Commercial Property",
        ),
        DocumentChunk(
            id=201,
            document_id=2,
            chunk_index=0,
            page_number=1,
            chunk_text="Commercial Auto Policy: Covers specified fleet vehicles for comprehensive and collision loss.",
            file_name="fleet_auto_policy.pdf",
            document_type="Auto Policy",
        ),
        DocumentChunk(
            id=202,
            document_id=2,
            chunk_index=1,
            page_number=2,
            chunk_text="Automobile Physical Damage Deductible: $1,000 comprehensive, $1,500 collision per vehicle.",
            file_name="fleet_auto_policy.pdf",
            document_type="Auto Policy",
        ),
    ]

    indexed_count = vector_store.add_chunks(test_chunks)
    assert indexed_count == 5
    assert vector_store.get_collection_count() == 5
    print(f"  -> Indexed {indexed_count} chunks with full metadata into ChromaDB.")

    # 4. Test Semantic Similarity Search
    print("\n[4/6] Testing similarity search and top-5 chunk retrieval...")
    results = vector_store.similarity_search(
        query="What is the property deductible?",
        top_k=5,
    )
    assert len(results) > 0, "Expected at least 1 matching chunk"
    assert len(results) <= 5, f"Expected <= 5 results, got {len(results)}"

    top_match = results[0]
    # Verify all required fields are present in the retrieved item
    assert "chunk_id" in top_match and top_match["chunk_id"] is not None
    assert "document_id" in top_match and top_match["document_id"] is not None
    assert "document_name" in top_match and top_match["document_name"] != ""
    assert "document_type" in top_match and top_match["document_type"] != ""
    assert "page_number" in top_match and top_match["page_number"] >= 1
    assert "text" in top_match and top_match["text"] != ""
    assert "similarity_score" in top_match and 0.0 <= top_match["similarity_score"] <= 1.0

    print(f"  -> Top result (Score: {top_match['similarity_score']}):")
    print(f"     Doc: '{top_match['document_name']}' | Page: {top_match['page_number']} | Chunk ID: {top_match['chunk_id']}")
    print(f"     Text: '{top_match['text'][:70]}...'")

    # 5. Test Metadata Filtering (by document_id and document_type)
    print("\n[5/6] Testing metadata filtering in ChromaDB...")
    # Filter by document_id = 2 (Auto Policy only)
    auto_results = vector_store.similarity_search(
        query="What is the deductible?",
        top_k=5,
        document_id=2,
    )
    assert all(r["document_id"] == 2 for r in auto_results), "Expected all results to have document_id=2"
    print(f"  -> Filter document_id=2 returned {len(auto_results)} chunks (all doc_id=2).")

    # Filter by document_type = "Commercial Property"
    property_results = vector_store.similarity_search(
        query="What is the coverage limit?",
        top_k=5,
        document_type="Commercial Property",
    )
    assert all(r["document_type"] == "Commercial Property" for r in property_results)
    print(f"  -> Filter document_type='Commercial Property' returned {len(property_results)} chunks.")

    # Filter with non-existent document ID -> expect empty results
    empty_results = vector_store.similarity_search(
        query="Coverage limit",
        top_k=5,
        document_id=9999,
    )
    assert len(empty_results) == 0
    print("  -> Filter non-existent document_id=9999 returned 0 results as expected.")

    # 6. Test Live HTTP POST /search Endpoint
    print("\n[6/6] Testing live HTTP POST /search endpoint via FastAPI & Uvicorn...")
    # Override vector store dependency in FastAPI app
    fastapi_app.dependency_overrides[get_vector_store_service] = lambda: vector_store

    server_thread = threading.Thread(
        target=lambda: __import__("uvicorn").run(fastapi_app, host="127.0.0.1", port=8004, log_level="error"),
        daemon=True,
    )
    server_thread.start()
    time.sleep(2)

    search_payload = {
        "query": "What is the deductible and coverage limit?",
        "top_k": 5,
        "document_type": "Commercial Property",
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8004/search",
        data=json.dumps(search_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"Expected 200 OK, got {resp.status}"
        resp_data = json.loads(resp.read().decode("utf-8"))
        assert resp_data["query"] == search_payload["query"]
        assert resp_data["total_results"] > 0
        assert len(resp_data["results"]) <= 5
        first_item = resp_data["results"][0]
        assert first_item["document_type"] == "Commercial Property"
        print(f"  -> HTTP 200 OK: query='{resp_data['query']}', returned {resp_data['total_results']} chunks.")
        print(f"     First chunk: [{first_item['document_name']} p.{first_item['page_number']}] score={first_item['similarity_score']}")

    cleanup_test_dir()
    print("\n" + "=" * 65)
    print("ALL PHASE 4 AUTOMATED TESTS PASSED SUCCESSFULLY (100% SUCCESS)!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
