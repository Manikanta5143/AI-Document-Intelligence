# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session

from app.aws.llm_service import BaseLLMService, get_llm_service
from app.database.session import get_db
from app.rag.rag_service import RAGService
from app.rag.vector_store import VectorStoreService, get_vector_store_service
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.query_repository import QueryRepository
from app.schemas.analysis import DocumentExtractionResponse, DocumentSummaryResponse
from app.schemas.document import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from app.schemas.rag import RAGAnswerResponse, RAGQuestionRequest
from app.services.analysis_service import DocumentAnalysisService
from app.services.document_service import DocumentService
from app.services.snowflake_analytics_service import BaseAnalyticsService, get_analytics_service

router = APIRouter(prefix="/documents", tags=["Documents"])


def get_document_repository(db: Session = Depends(get_db)) -> DocumentRepository:
    """Dependency provider for DocumentRepository with active DB session."""
    return DocumentRepository(db=db)


def get_chunk_repository(db: Session = Depends(get_db)) -> ChunkRepository:
    """Dependency provider for ChunkRepository with active DB session."""
    return ChunkRepository(db=db)


def get_query_repository(db: Session = Depends(get_db)) -> QueryRepository:
    """Dependency provider for QueryRepository with active DB session."""
    return QueryRepository(db=db)


def get_document_service(
    doc_repo: DocumentRepository = Depends(get_document_repository),
    chunk_repo: ChunkRepository = Depends(get_chunk_repository),
    vector_store: VectorStoreService = Depends(get_vector_store_service),
    analytics_service: BaseAnalyticsService = Depends(get_analytics_service),
) -> DocumentService:
    """Dependency provider for DocumentService with injected repositories, vector store, and analytics."""
    return DocumentService(
        document_repository=doc_repo,
        chunk_repository=chunk_repo,
        vector_store_service=vector_store,
        analytics_service=analytics_service,
    )


def get_analysis_service(
    doc_repo: DocumentRepository = Depends(get_document_repository),
    chunk_repo: ChunkRepository = Depends(get_chunk_repository),
    llm_service: BaseLLMService = Depends(get_llm_service),
) -> DocumentAnalysisService:
    """Dependency provider for DocumentAnalysisService keeping LLM logic out of routes."""
    return DocumentAnalysisService(
        document_repository=doc_repo,
        chunk_repository=chunk_repo,
        llm_service=llm_service,
    )


def get_rag_service(
    vector_store: VectorStoreService = Depends(get_vector_store_service),
    llm_service: BaseLLMService = Depends(get_llm_service),
    query_repo: QueryRepository = Depends(get_query_repository),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    analytics_service: BaseAnalyticsService = Depends(get_analytics_service),
) -> RAGService:
    """Dependency provider for RAGService coordinating vector retrieval, LLM, MySQL persistence, and analytics."""
    return RAGService(
        vector_store=vector_store,
        llm_service=llm_service,
        query_repository=query_repo,
        document_repository=doc_repo,
        analytics_service=analytics_service,
    )


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Ingest Insurance Document (PDF)",
    description=(
        "Uploads an insurance PDF file, stores it in AWS S3, creates document metadata in MySQL, "
        "extracts text per page using PyMuPDF, cleans text, splits it into chunks with configurable "
        "size/overlap, saves chunk metadata to MySQL, and indexes vectors in ChromaDB."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="Insurance PDF file (e.g. policy, claim, endorsement)"),
    document_type: str = Form("Policy", description="Document type (e.g. Policy, Claim, Endorsement)"),
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """Endpoint for document upload and ingestion pipeline."""
    file_bytes = await file.read()
    return service.process_and_upload_document(
        file_name=file.filename or "uploaded_document.pdf",
        file_bytes=file_bytes,
        document_type=document_type,
    )


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register Insurance Document Metadata",
    description="Registers initial metadata for an insurance document (file name and document type) in the database with PENDING status.",
)
def create_document(
    payload: DocumentCreate,
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Create document metadata in MySQL."""
    return service.create_document_metadata(payload)


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List All Documents",
    description="Returns a paginated list of insurance documents stored in the database.",
)
def list_documents(
    skip: int = Query(0, ge=0, description="Offset number of records to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of records to return"),
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """List documents with pagination."""
    return service.list_documents(skip=skip, limit=limit)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Document by ID",
    description="Retrieve specific document metadata by its primary key ID.",
)
def get_document(
    document_id: int,
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Get single document by ID."""
    return service.get_document_by_id(document_id=document_id)


@router.post(
    "/{document_id}/summary",
    response_model=DocumentSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Insurance Document Summary",
    description=(
        "Analyzes the processed text chunks of an insurance document and generates "
        "an executive narrative summary and key bullet points using Amazon Nova Lite."
    ),
)
def summarize_document(
    document_id: int,
    analysis_service: DocumentAnalysisService = Depends(get_analysis_service),
) -> DocumentSummaryResponse:
    """Generate document summary using Amazon Nova Lite without hardcoded LLM logic in route."""
    return analysis_service.summarize_document(document_id=document_id)


@router.post(
    "/{document_id}/extract",
    response_model=DocumentExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Structured Insurance Data",
    description=(
        "Extracts structured insurance attributes (policy type, coverage limits, exclusions, "
        "deductibles, important conditions, and risk indicators) into validated JSON using Amazon Nova Lite."
    ),
)
def extract_document_info(
    document_id: int,
    analysis_service: DocumentAnalysisService = Depends(get_analysis_service),
) -> DocumentExtractionResponse:
    """Extract structured insurance data using Amazon Nova Lite without hallucination."""
    return analysis_service.extract_structured_info(document_id=document_id)


@router.post(
    "/{document_id}/ask",
    response_model=RAGAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask Question About Insurance Document (RAG)",
    description=(
        "Executes full Retrieval-Augmented Generation (RAG) against a specific insurance document. "
        "Embeds the question, retrieves top-k relevant chunks from ChromaDB, prompts Amazon Nova Lite "
        "with strict grounding instructions, saves the Q&A in MySQL, and returns the answer with source attribution."
    ),
)
def ask_document_question(
    document_id: int,
    payload: RAGQuestionRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGAnswerResponse:
    """Execute complete RAG pipeline for an insurance document."""
    return rag_service.ask_question(
        document_id=document_id,
        question=payload.question,
        top_k=payload.top_k,
        temperature=payload.temperature,
    )
