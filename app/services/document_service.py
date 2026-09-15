from datetime import datetime
import io
from typing import Any, List, Optional
# pyrefly: ignore [missing-import]
from fastapi import HTTPException, status
    
from app.aws.s3_service import S3Service
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.analytics import DocumentAnalyticsRecord
from app.schemas.chunk import ChunkResponse
from app.schemas.document import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from app.services.pdf_extractor import PDFExtractor
from app.services.snowflake_analytics_service import BaseAnalyticsService
from app.services.text_chunker import TextChunker
from app.utils.logger import setup_logger

logger = setup_logger("document_service")


class DocumentService:
    """Service layer coordinating business logic, S3 operations, text extraction,
    chunking, repository persistence, and analytics sync for documents."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        chunk_repository: Optional[ChunkRepository] = None,
        s3_service: Optional[S3Service] = None,
        pdf_extractor: Optional[PDFExtractor] = None,
        text_chunker: Optional[TextChunker] = None,
        vector_store_service: Optional[Any] = None,
        analytics_service: Optional[BaseAnalyticsService] = None,
    ):
        self._document_repo = document_repository
        self._chunk_repo = chunk_repository
        self._s3_service = s3_service or S3Service()
        self._pdf_extractor = pdf_extractor or PDFExtractor()
        self._text_chunker = text_chunker or TextChunker()
        self._vector_store_service = vector_store_service
        self._analytics_service = analytics_service

    def create_document_metadata(self, payload: DocumentCreate) -> DocumentResponse:
        """Create initial metadata record for an insurance document."""
        logger.info("Registering metadata for document '%s' (type='%s')", payload.file_name, payload.document_type)
        new_doc = Document(
            file_name=payload.file_name,
            document_type=payload.document_type,
            s3_key=payload.s3_key,
            status="PENDING",
        )
        saved_doc = self._document_repo.create(new_doc)
        return DocumentResponse.model_validate(saved_doc)

    def get_document_by_id(self, document_id: int) -> DocumentResponse:
        """Retrieve a specific document by its database ID."""
        document = self._document_repo.get_by_id(document_id)
        if not document:
            logger.warning("Document with ID %s not found.", document_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} was not found.",
            )
        return DocumentResponse.model_validate(document)

    def list_documents(self, skip: int = 0, limit: int = 100) -> DocumentListResponse:
        """Retrieve a paginated list of registered documents."""
        total = self._document_repo.count_all()
        documents = self._document_repo.get_all(skip=skip, limit=limit)
        items = [DocumentResponse.model_validate(doc) for doc in documents]
        return DocumentListResponse(total=total, documents=items)

    def process_and_upload_document(
        self,
        file_name: str,
        file_bytes: bytes,
        document_type: str,
    ) -> DocumentUploadResponse:
        """Execute the full Phase 3 document ingestion pipeline:
        1. Validate file format and integrity.
        2. Upload binary to AWS S3 via boto3.
        3. Persist document metadata in MySQL with PROCESSING status.
        4. Extract page-aware text using PyMuPDF and clean text.
        5. Split extracted text into overlapping chunks.
        6. Persist chunk records in MySQL.
        7. Update document status to PROCESSED.

        Args:
            file_name: Name of the uploaded file.
            file_bytes: Raw binary content of the PDF.
            document_type: Insurance document category.

        Raises:
            HTTPException: On validation failure, S3 failure, or extraction errors.

        Returns:
            DocumentUploadResponse: Detailed upload summary and chunk metadata.
        """
        # 1. File Validation
        if not file_name or not file_name.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files (.pdf) are supported.",
            )

        if not file_bytes or len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        if not file_bytes.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file does not have a valid PDF header.",
            )

        logger.info(
            "Starting Phase 3 document processing for '%s' (%d bytes, type='%s')",
            file_name,
            len(file_bytes),
            document_type,
        )

        # 2. Upload to S3
        s3_key = self._s3_service.generate_s3_key(file_name)
        try:
            self._s3_service.upload_fileobj(
                file_obj=io.BytesIO(file_bytes),
                s3_key=s3_key,
                content_type="application/pdf",
            )
        except ValueError as ve:
            logger.error("S3 configuration error: %s", ve)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(ve),
            ) from ve
        except Exception as e:
            logger.error("Failed to upload document to S3: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"AWS S3 Upload failed: {str(e)}",
            ) from e

        # 3. Create MySQL document record
        initial_doc = Document(
            file_name=file_name,
            document_type=document_type,
            s3_key=s3_key,
            status="PROCESSING",
        )
        saved_doc = self._document_repo.create(initial_doc)

        try:
            # 4. Extract PDF text using PyMuPDF
            pages_data = self._pdf_extractor.extract_pages(file_bytes)

            # 5. Split text into chunks
            chunk_dicts = self._text_chunker.chunk_document_pages(pages_data)

            # 6. Save chunk metadata to MySQL
            chunk_models = [
                DocumentChunk(
                    document_id=saved_doc.id,
                    chunk_index=c["chunk_index"],
                    page_number=c["page_number"],
                    chunk_text=c["chunk_text"],
                    file_name=saved_doc.file_name,
                    document_type=saved_doc.document_type,
                )
                for c in chunk_dicts
            ]

            persisted_chunks: List[DocumentChunk] = []
            if self._chunk_repo and chunk_models:
                persisted_chunks = self._chunk_repo.create_bulk(chunk_models)
            elif not self._chunk_repo and chunk_models:
                logger.warning("ChunkRepository not provided to service; skipping chunk persistence.")

            # Index chunks in ChromaDB vector store if service is configured
            if self._vector_store_service and persisted_chunks:
                try:
                    self._vector_store_service.add_chunks(persisted_chunks)
                    logger.info("Indexed %d chunks into ChromaDB vector store.", len(persisted_chunks))
                except Exception as err:
                    logger.warning("Vector store indexing warning: %s", str(err))

            # 7. Update document status to PROCESSED
            now_ts = datetime.utcnow()
            self._document_repo.update_status(
                document_id=saved_doc.id,
                status="PROCESSED",
                processed_at=now_ts,
            )

            # 8. Sync initial document metrics to Snowflake analytics if configured
            if self._analytics_service:
                try:
                    proc_time = 1.0
                    if saved_doc.created_at:
                        delta = (now_ts - saved_doc.created_at).total_seconds()
                        proc_time = max(round(delta, 2), 0.1)
                    analytics_rec = DocumentAnalyticsRecord(
                        document_id=saved_doc.id,
                        document_type=saved_doc.document_type,
                        processing_time=proc_time,
                        chunk_count=len(persisted_chunks),
                        query_count=0,
                        risk_category="LOW",
                        created_at=saved_doc.created_at,
                    )
                    self._analytics_service.sync_document_analytics(analytics_rec)
                    logger.info("Automatically synced document ID %d metrics to Snowflake.", saved_doc.id)
                except Exception as sync_err:
                    logger.warning("Snowflake auto-sync warning for doc ID %d: %s", saved_doc.id, str(sync_err))

            chunk_responses = [
                ChunkResponse(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    text=chunk.chunk_text,
                    file_name=chunk.file_name,
                    document_type=chunk.document_type,
                )
                for chunk in persisted_chunks
            ]

            logger.info(
                "Document ID=%d processing completed successfully (%d pages, %d chunks saved).",
                saved_doc.id,
                len(pages_data),
                len(persisted_chunks),
            )

            return DocumentUploadResponse(
                document_id=saved_doc.id,
                file_name=saved_doc.file_name,
                document_type=saved_doc.document_type,
                s3_key=saved_doc.s3_key or s3_key,
                status="PROCESSED",
                total_pages=len(pages_data),
                total_chunks=len(persisted_chunks),
                chunks=chunk_responses,
            )

        except Exception as e:
            logger.error(
                "Error processing text or saving chunks for document ID=%d: %s",
                saved_doc.id,
                str(e),
            )
            # Mark document as FAILED in database
            self._document_repo.update_status(document_id=saved_doc.id, status="FAILED")
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document processing failed: {str(e)}",
            ) from e
