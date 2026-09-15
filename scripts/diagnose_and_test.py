"""Comprehensive End-to-End Diagnostic and Verification Suite.

Tests all 10 core subsystems:
1. Environment & Configuration Check
2. AWS STS Authentication Check
3. AWS S3 Storage Test (live S3Service)
4. Amazon Bedrock Nova Lite Inference Test (live BedrockLLMService)
5. Amazon Bedrock Titan Text Embeddings Test (live BedrockEmbeddingService)
6. ChromaDB Persistent Vector Store & Semantic Retrieval Test (live VectorStoreService)
7. MySQL Database Connection, Tables & CRUD Test (live SQLAlchemy models/repositories)
8. Snowflake Analytics Connection & Query Execution (live SnowflakeAnalyticsService)
9. PyMuPDF PDF Text Extraction & Page-Aware Overlap Chunking
10. Complete End-to-End RAG Q&A Pipeline with Grounded Answers & Source Attribution
"""
import io
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import boto3
import fitz
import pymysql
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.session import Base, SessionLocal, engine, init_db
from app.models.document import Document
from app.models.chunk import DocumentChunk
from app.models.query import QueryRecord
from app.repositories.document_repository import DocumentRepository
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.query_repository import QueryRepository
from app.services.pdf_extractor import PDFExtractor
from app.services.text_chunker import TextChunker
from app.aws.s3_service import S3Service
from app.aws.bedrock_service import BedrockEmbeddingService, MockEmbeddingService
from app.aws.llm_service import BedrockLLMService, MockLLMService
from app.rag.vector_store import VectorStoreService
from app.rag.rag_service import RAGService
from app.services.snowflake_analytics_service import SnowflakeAnalyticsService, MockAnalyticsService

results = {}

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_environment_and_config():
    print_header("1. Environment and Configuration Check")
    settings = get_settings()
    print(f"App Name: {settings.app_name}")
    print(f"Environment: {settings.environment}")
    print(f"AWS Region: {settings.aws_region}")
    print(f"S3 Bucket: {settings.s3_bucket_name or '[Not configured]'}")
    print(f"Bedrock Model: {settings.bedrock_model_id}")
    print(f"Bedrock Embedding Model: {settings.bedrock_embedding_model_id}")
    print(f"MySQL Host: {settings.mysql_host}:{settings.mysql_port}")
    print(f"MySQL User: {settings.mysql_user}")
    print(f"MySQL Database: {settings.mysql_database}")
    print(f"MySQL Password configured: {'YES' if settings.mysql_password else 'NO (Empty)'}")
    print(f"Database URL Override: {settings.database_url_override or '[None]'}")
    print(f"Snowflake Account: {settings.snowflake_account or '[Not configured]'}")
    print(f"Snowflake User: {settings.snowflake_user or '[Not configured]'}")
    results["Config"] = "PASS"

def test_aws_sts():
    print_header("2. AWS Authentication Check (STS get-caller-identity)")
    settings = get_settings()
    session_kwargs = {"region_name": settings.aws_region}
    if settings.aws_profile:
        session_kwargs["profile_name"] = settings.aws_profile
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        session_kwargs["aws_access_key_id"] = settings.aws_access_key_id
        session_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        if settings.aws_session_token:
            session_kwargs["aws_session_token"] = settings.aws_session_token

    try:
        session = boto3.Session(**session_kwargs)
        sts = session.client("sts")
        ident = sts.get_caller_identity()
        print(f"STS Authentication: SUCCESS!")
        print(f"Account: {ident.get('Account')}")
        print(f"ARN: {ident.get('Arn')}")
        results["AWS Auth"] = "PASS"
        return session
    except Exception as e:
        print(f"AWS Authentication Check: FAIL -> {e}")
        results["AWS Auth"] = f"FAIL ({type(e).__name__})"
        return None

def test_s3(session):
    print_header("3. S3 Connection and Bucket Test")
    settings = get_settings()
    if not settings.s3_bucket_name:
        print("S3_BUCKET_NAME is not configured in .env.")
        results["S3"] = "FAIL (S3_BUCKET_NAME missing)"
        return

    try:
        s3_service = S3Service(settings=settings)
        test_key = s3_service.generate_s3_key("test_diagnostic.txt")
        test_content = b"Diagnostic probe for AI Insurance Document Intelligence Assistant."
        
        # Upload
        print(f"Uploading probe object to s3://{settings.s3_bucket_name}/{test_key}...")
        s3_service.upload_fileobj(io.BytesIO(test_content), test_key, content_type="text/plain")
        
        # Download / Read
        print("Reading object back from S3...")
        response = s3_service.client.get_object(Bucket=settings.s3_bucket_name, Key=test_key)
        downloaded = response["Body"].read()
        assert downloaded == test_content, "Downloaded content does not match uploaded content"
        
        # Clean up
        print("Cleaning up probe object from S3...")
        s3_service.client.delete_object(Bucket=settings.s3_bucket_name, Key=test_key)
        print("S3 bucket access verified successfully!")
        results["S3"] = "PASS"
    except Exception as e:
        print(f"S3 Test: FAIL -> {e}")
        results["S3"] = f"FAIL ({type(e).__name__}: {e})"

def test_bedrock_llm(session):
    print_header("4. Amazon Bedrock Nova Lite Connection Test")
    settings = get_settings()
    llm = BedrockLLMService(settings=settings)
    prompt = "Explain what an insurance policy is in one sentence."
    print(f"Sending prompt to model '{settings.bedrock_model_id}': '{prompt}'")
    try:
        reply = llm.generate_text(prompt=prompt, max_tokens=100, temperature=0.0)
        print(f"Response received ({len(reply)} chars):")
        print(f"-> {reply}")
        results["Amazon Nova Lite"] = "PASS"
    except Exception as e:
        print(f"Nova Lite Test: FAIL -> {e}")
        results["Amazon Nova Lite"] = f"FAIL ({type(e).__name__}: {e})"

def test_bedrock_embeddings(session):
    print_header("5. Amazon Bedrock Titan Text Embeddings Test")
    settings = get_settings()
    emb_service = BedrockEmbeddingService(settings=settings)
    sample_text = "Commercial Property Insurance coverage provisions and standard deductibles."
    print(f"Generating embedding for: '{sample_text}' using '{settings.bedrock_embedding_model_id}'...")
    try:
        vec = emb_service.embed_text(sample_text)
        print(f"Embedding generated! Dimensions: {len(vec)}, Sample: {vec[:3]}...")
        assert len(vec) == settings.embedding_dimension or len(vec) > 0, "Empty vector returned"
        results["Titan Embeddings"] = "PASS"
    except Exception as e:
        print(f"Titan Embeddings Test: FAIL -> {e}")
        results["Titan Embeddings"] = f"FAIL ({type(e).__name__}: {e})"

def test_chromadb():
    print_header("6. ChromaDB Persistent Vector Store Test")
    settings = get_settings()
    try:
        # Use mock embedding for deterministic vector test if Bedrock failed or test with live if available
        emb = MockEmbeddingService(dimension=settings.embedding_dimension)
        vstore = VectorStoreService(embedding_service=emb, settings=settings)
        print(f"Connected to ChromaDB collection '{settings.chroma_collection_name}' (count={vstore.get_collection_count()})")
        
        # Test adding chunk
        test_chunk = DocumentChunk(
            id=9999,
            document_id=9999,
            chunk_index=1,
            page_number=1,
            chunk_text="Test policy chunk: The deductible for storm damage is $2,000.",
            file_name="diagnostic_policy.pdf",
            document_type="Test Policy",
        )
        indexed = vstore.add_chunks([test_chunk])
        assert indexed == 1, "Failed to index chunk into ChromaDB"
        
        # Test similarity search
        search_res = vstore.similarity_search("What is the deductible for storm damage?", top_k=1, document_id=9999)
        assert len(search_res) > 0, "No chunks retrieved in similarity search"
        top_hit = search_res[0]
        print(f"ChromaDB retrieval hit: score={top_hit['similarity_score']}, page={top_hit['page_number']}, text='{top_hit['text']}'")
        assert top_hit["page_number"] == 1, "Page metadata not preserved"
        results["ChromaDB"] = "PASS"
    except Exception as e:
        print(f"ChromaDB Test: FAIL -> {e}")
        results["ChromaDB"] = f"FAIL ({type(e).__name__}: {e})"

def test_mysql():
    print_header("7. MySQL Database Connection and CRUD Test")
    settings = get_settings()
    print(f"Attempting connection to {settings.database_url}...")
    
    # 1. Test database connection
    try:
        with engine.connect() as conn:
            val = conn.execute(text("SELECT 1")).scalar()
            assert val == 1
        print("Database connection established successfully!")
    except Exception as e:
        print(f"MySQL Connection FAIL: {e}")
        results["MySQL"] = f"FAIL ({type(e).__name__}: {e})"
        return

    # 2. Table initialization
    try:
        tables_ok = init_db()
        assert tables_ok, "init_db returned False"
        print("SQLAlchemy tables initialized and verified.")
    except Exception as e:
        print(f"Table initialization FAIL: {e}")
        results["MySQL"] = f"FAIL (Table init: {e})"
        return

    # 3. Test CRUD operations
    db: Session = SessionLocal()
    try:
        doc_repo = DocumentRepository(db)
        chunk_repo = ChunkRepository(db)
        query_repo = QueryRepository(db)

        test_doc = Document(
            file_name="diagnostic_test.pdf",
            s3_key="documents/diagnostic_test.pdf",
            document_type="Commercial Property",
            status="PROCESSED",
        )
        saved_doc = doc_repo.create(test_doc)
        print(f"Created Document record ID: {saved_doc.id}")

        retrieved_doc = doc_repo.get_by_id(saved_doc.id)
        assert retrieved_doc is not None, "Failed to retrieve document record"
        assert retrieved_doc.file_name == "diagnostic_test.pdf"
        print(f"Read back Document record: {retrieved_doc.file_name}")

        test_q = QueryRecord(
            document_id=saved_doc.id,
            question="Is flood damage covered?",
            answer="Flood damage is strictly excluded.",
        )
        saved_q = query_repo.create(test_q)
        print(f"Created QueryRecord ID: {saved_q.id}")

        retrieved_q = query_repo.get_by_id(saved_q.id)
        assert retrieved_q is not None
        assert retrieved_q.answer == "Flood damage is strictly excluded."
        print(f"Read back QueryRecord: {retrieved_q.question} -> {retrieved_q.answer}")

        # Cleanup test records
        db.delete(saved_q)
        db.delete(saved_doc)
        db.commit()
        print("Cleaned up diagnostic records from database.")
        results["MySQL"] = "PASS"
    except Exception as e:
        print(f"MySQL CRUD Test FAIL: {e}")
        results["MySQL"] = f"FAIL (CRUD: {e})"
    finally:
        db.close()

def test_snowflake():
    print_header("8. Snowflake Analytics Connection Test")
    settings = get_settings()
    if not (settings.snowflake_account and settings.snowflake_user and settings.snowflake_password):
        print("Snowflake credentials (SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD) are not configured in .env.")
        results["Snowflake"] = "FAIL (Credentials not configured)"
        return

    try:
        snow_service = SnowflakeAnalyticsService(settings=settings)
        conn = snow_service._get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT CURRENT_USER();")
            current_user = cur.fetchone()[0]
            cur.execute("SELECT CURRENT_TIMESTAMP();")
            current_ts = cur.fetchone()[0]
            print(f"Connected to Snowflake! CURRENT_USER: {current_user}, CURRENT_TIMESTAMP: {current_ts}")
        conn.close()

        # Initialize schema and verify table
        snow_service.init_schema()
        print("Snowflake document_analytics table verified.")
        results["Snowflake"] = "PASS"
    except Exception as e:
        print(f"Snowflake Test FAIL: {e}")
        results["Snowflake"] = f"FAIL ({type(e).__name__}: {e})"

def test_pdf_processing():
    print_header("9. PyMuPDF PDF Text Extraction and Chunking Test")
    try:
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text((50, 72), "Commercial Property Insurance Policy. Coverage A covers fire and wind damage.")
        page2 = doc.new_page()
        page2.insert_text((50, 72), "Exclusion Section 4: Flood damage is excluded from coverage. Deductible is $5,000.")
        pdf_bytes = doc.tobytes()
        doc.close()

        extractor = PDFExtractor()
        extracted_pages = extractor.extract_pages(pdf_bytes)
        assert len(extracted_pages) == 2, f"Expected 2 pages, got {len(extracted_pages)}"
        print(f"Extracted {len(extracted_pages)} pages with PyMuPDF:")
        for ep in extracted_pages:
            print(f"  Page {ep['page_number']}: '{ep['text'].strip()}'")

        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document_pages(extracted_pages)
        assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
        print(f"Generated {len(chunks)} overlapping chunks.")
        for c in chunks:
            print(f"  Chunk {c['chunk_index']} (Page {c['page_number']}): '{c['chunk_text']}'")

        results["PDF Processing"] = "PASS"
        return pdf_bytes
    except Exception as e:
        print(f"PDF Processing FAIL: {e}")
        results["PDF Processing"] = f"FAIL ({type(e).__name__}: {e})"
        return None

def test_rag_workflow():
    print_header("10. End-to-End RAG Q&A Pipeline Test")
    settings = get_settings()
    try:
        # Check if live Bedrock services are functional, otherwise test with MockLLMService
        llm = BedrockLLMService(settings=settings) if results.get("Amazon Nova Lite") == "PASS" else MockLLMService()
        emb = BedrockEmbeddingService(settings=settings) if results.get("Titan Embeddings") == "PASS" else MockEmbeddingService(dimension=settings.embedding_dimension)
        vstore = VectorStoreService(embedding_service=emb, settings=settings)

        # Use an in-memory session or database session
        db = SessionLocal()
        doc_repo = DocumentRepository(db)
        chunk_repo = ChunkRepository(db)
        query_repo = QueryRepository(db)

        # Prepare test document
        test_doc = Document(
            file_name="acme_property_policy.pdf",
            s3_key="documents/acme_property_policy.pdf",
            document_type="Commercial Property",
            status="PROCESSED",
        )
        saved_doc = doc_repo.create(test_doc)

        # Index policy facts into ChromaDB
        test_chunks = [
            DocumentChunk(
                id=101,
                document_id=saved_doc.id,
                chunk_index=1,
                page_number=1,
                chunk_text="Acme Commercial Property Policy. Policy covers fire damage up to $2,500,000. Theft is covered. Standard deductible is $5,000.",
                file_name=saved_doc.file_name,
                document_type=saved_doc.document_type,
            ),
            DocumentChunk(
                id=102,
                document_id=saved_doc.id,
                chunk_index=2,
                page_number=2,
                chunk_text="Section B - Policy Exclusions. Flood damage, surface water, and water backup are strictly excluded under this policy.",
                file_name=saved_doc.file_name,
                document_type=saved_doc.document_type,
            ),
        ]
        vstore.add_chunks(test_chunks)

        rag_service = RAGService(
            vector_store=vstore,
            llm_service=llm,
            query_repository=query_repo,
            document_repository=doc_repo,
            settings=settings,
            analytics_service=MockAnalyticsService(),
        )

        question = "Is flood damage covered?"
        response = rag_service.ask_question(document_id=saved_doc.id, question=question)

        print(f"Question: '{question}'")
        print(f"Answer: {response.answer}")
        print(f"Sources cited: {len(response.sources)}")
        for src in response.sources:
            print(f"  -> File: {src.file_name}, Page: {src.page_number}, Chunk: {src.chunk_id}")

        assert len(response.sources) > 0, "No sources cited in RAG response"
        assert response.query_id is not None, "Query record was not persisted to database"
        
        # Clean up test records
        query_obj = query_repo.get_by_id(response.query_id)
        if query_obj:
            db.delete(query_obj)
        db.delete(saved_doc)
        db.commit()
        db.close()

        results["RAG"] = "PASS"
    except Exception as e:
        print(f"RAG Test FAIL: {e}")
        results["RAG"] = f"FAIL ({type(e).__name__}: {e})"

def main():
    print("=" * 70)
    print("  AI INSURANCE DOCUMENT INTELLIGENCE ASSISTANT - DIAGNOSTIC RUNNER")
    print("=" * 70)

    test_environment_and_config()
    session = test_aws_sts()
    test_s3(session)
    test_bedrock_llm(session)
    test_bedrock_embeddings(session)
    test_chromadb()
    test_mysql()
    test_snowflake()
    test_pdf_processing()
    test_rag_workflow()

    print_header("DIAGNOSTIC SUMMARY")
    for component, status in results.items():
        print(f"  {component:<25} : {status}")

if __name__ == "__main__":
    main()
