from typing import List, Optional
from fastapi import HTTPException, status

from app.aws.llm_service import BaseLLMService, get_llm_service
from app.config import Settings, get_settings
from app.models.query import QueryRecord
from app.rag.vector_store import VectorStoreService, get_vector_store_service
from app.repositories.document_repository import DocumentRepository
from app.repositories.query_repository import QueryRepository
from app.schemas.rag import RAGAnswerResponse, SourceMetadata
from app.services.snowflake_analytics_service import BaseAnalyticsService
from app.utils.logger import setup_logger

logger = setup_logger("rag_service")


class RAGService:
    """Coordinates the end-to-end Retrieval-Augmented Generation pipeline:
    1. Validates document presence in MySQL.
    2. Embeds user question and performs similarity search in ChromaDB scoped to document.
    3. Assembles grounded context with page numbers and chunk metadata.
    4. Invokes Amazon Nova Lite with anti-hallucination, no-chain-of-thought system prompts.
    5. Formats answer with source attributions.
    6. Persists Q&A audit record in MySQL queries table.
    7. Updates analytical query counts in Snowflake.
    """

    def __init__(
        self,
        vector_store: VectorStoreService,
        llm_service: BaseLLMService,
        query_repository: QueryRepository,
        document_repository: DocumentRepository,
        settings: Optional[Settings] = None,
        analytics_service: Optional[BaseAnalyticsService] = None,
    ):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.query_repo = query_repository
        self.doc_repo = document_repository
        self.settings = settings or get_settings()
        self.analytics_service = analytics_service

    def ask_question(
        self,
        document_id: int,
        question: str,
        top_k: Optional[int] = None,
    ) -> RAGAnswerResponse:
        """Execute RAG question-answering workflow on an insurance document.

        Args:
            document_id: Database ID of the target document.
            question: User's insurance question.
            top_k: Optional override for number of chunks to retrieve (defaults to config).

        Raises:
            HTTPException: 404 if document is not found, 400 for empty questions, 500 for failures.

        Returns:
            RAGAnswerResponse: Grounded answer, source list, document_id, and query_id.
        """
        if not question or not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty.",
            )

        clean_question = question.strip()

        # 1. Verify document existence
        doc = self.doc_repo.get_by_id(document_id)
        if not doc:
            logger.warning("RAG query attempted for non-existent document ID %d.", document_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} was not found.",
            )

        k = top_k or self.settings.rag_top_k
        logger.info(
            "Executing RAG pipeline for document ID %d ('%s'), top_k=%d, question='%s'",
            document_id,
            doc.file_name,
            k,
            clean_question,
        )

        try:
            # 2. Retrieve relevant chunks strictly scoped to this document
            retrieved_chunks = self.vector_store.similarity_search(
                query=clean_question,
                top_k=k,
                document_id=document_id,
            )

            sources: List[SourceMetadata] = []
            context_blocks: List[str] = []

            for idx, chunk in enumerate(retrieved_chunks, 1):
                file_name = chunk.get("document_name") or doc.file_name
                page_num = chunk.get("page_number", 1)
                chunk_id = chunk.get("chunk_id", idx)
                chunk_text = chunk.get("text", "").strip()

                if chunk_text:
                    context_blocks.append(
                        f"[Excerpt {idx} | Page {page_num} | Chunk ID: {chunk_id}]\n{chunk_text}"
                    )
                    sources.append(
                        SourceMetadata(
                            file_name=file_name,
                            page_number=page_num,
                            chunk_id=chunk_id,
                        )
                    )

            if not context_blocks:
                logger.info("No matching chunks found in vector store for document ID %d.", document_id)
                answer = "The document does not provide enough information to answer this question."
            else:
                retrieved_context = "\n\n".join(context_blocks)

                # 3. Grounded Prompt Construction
                system_instruction = (
                    "You are an insurance document assistant.\n"
                    "Answer only from the supplied context.\n"
                    "Do not invent policy details.\n"
                    "If the answer cannot be determined from the context, "
                    "clearly state that the document does not provide enough information.\n"
                    "Do not make unsupported assumptions.\n"
                    "Do not expose internal reasoning or chain-of-thought.\n"
                    "Provide a direct, clear, professional answer citing page numbers when relevant."
                )

                user_prompt = (
                    f"Document Name: {doc.file_name}\n"
                    f"Document Category: {doc.document_type}\n\n"
                    f"Retrieved Document Context:\n"
                    f"{retrieved_context}\n\n"
                    f"User Question: {clean_question}\n\n"
                    f"Answer:"
                )

                # 4. Invoke Amazon Nova Lite
                answer = self.llm_service.generate_text(
                    prompt=user_prompt,
                    system_prompt=system_instruction,
                    temperature=0.0,  # Zero-temperature for deterministic grounded answers
                )

            # 5. Persist Query and Answer in MySQL
            query_record = QueryRecord(
                document_id=document_id,
                question=clean_question,
                answer=answer.strip(),
            )
            saved_query = self.query_repo.create(query_record)
            logger.info("Persisted Q&A exchange ID=%d for document ID=%d.", saved_query.id, document_id)

            # 6. Increment analytics query engagement metric in Snowflake
            if self.analytics_service:
                try:
                    self.analytics_service.record_query_event(document_id)
                except Exception as a_err:
                    logger.warning("Snowflake query engagement metric warning: %s", str(a_err))

            return RAGAnswerResponse(
                answer=answer.strip(),
                sources=sources,
                document_id=document_id,
                query_id=saved_query.id,
            )

        except Exception as e:
            logger.error("RAG pipeline failed for document ID %d: %s", document_id, str(e))
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"RAG question-answering failed: {str(e)}",
            ) from e
