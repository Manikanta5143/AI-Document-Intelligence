import json
import re
from typing import List, Tuple
from fastapi import HTTPException, status

from app.aws.llm_service import BaseLLMService, get_llm_service
from app.models.document import Document
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.analysis import (
    DocumentExtractionResponse,
    DocumentSummaryResponse,
    StructuredInsuranceExtraction,
)
from app.utils.logger import setup_logger

logger = setup_logger("analysis_service")


class DocumentAnalysisService:
    """Service layer orchestrating document text aggregation, summarization,
    and structured insurance information extraction using Amazon Nova Lite."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        llm_service: BaseLLMService = None,
    ):
        self._doc_repo = document_repository
        self._chunk_repo = chunk_repository
        self._llm_service = llm_service or get_llm_service()

    def _assemble_document_text(self, document_id: int) -> Tuple[Document, str]:
        """Retrieve document and stitch chunks in sequential order with page headers."""
        doc = self._doc_repo.get_by_id(document_id)
        if not doc:
            logger.warning("Analysis requested for non-existent document ID %d.", document_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} was not found.",
            )

        chunks = self._chunk_repo.get_by_document_id(document_id)
        if not chunks:
            logger.warning("Document ID %d has no processed chunks.", document_id)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document '{doc.file_name}' (ID: {document_id}) has no extracted text chunks available.",
            )

        # Stitch text chronologically with page demarcation
        sections: List[str] = []
        current_page = None
        for chunk in chunks:
            if chunk.page_number != current_page:
                current_page = chunk.page_number
                sections.append(f"\n--- Page {current_page} ---")
            sections.append(chunk.chunk_text)

        assembled_text = "\n".join(sections).strip()
        return doc, assembled_text

    def summarize_document(self, document_id: int) -> DocumentSummaryResponse:
        """Generate an executive summary and key takeaways for an insurance document.

        Args:
            document_id: Database primary key of the document.

        Returns:
            DocumentSummaryResponse: Executive narrative and bulleted highlights.
        """
        doc, full_text = self._assemble_document_text(document_id)
        logger.info("Generating summary for document ID %d ('%s')...", document_id, doc.file_name)

        system_prompt = (
            "You are an expert insurance document analyst. Your role is to provide clear, "
            "factual, and professional summaries of insurance policies, claims, and endorsements. "
            "Base your answer strictly on the provided document text without inventing or assuming unstated details."
        )

        user_prompt = (
            f"Provide an executive summary and key bulleted takeaways for the insurance document below.\n\n"
            f"Document Name: {doc.file_name}\n"
            f"Document Type: {doc.document_type}\n\n"
            f"Document Content:\n{full_text}\n\n"
            f"Instructions:\n"
            f"1. Executive Summary: Provide 1-2 paragraphs summarizing the document scope, named parties, and main coverage.\n"
            f"2. Key Takeaways: Provide 3-6 concise bullet points highlighting specific coverage limits, deductibles, exclusions, or notable terms.\n"
            f"3. Strictly report only facts present in the text above."
        )

        try:
            model_id = getattr(self._llm_service, "model_id", "amazon.nova-lite-v1:0")
            raw_response = self._llm_service.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )

            # Parse bullet points from response
            lines = raw_response.split("\n")
            key_points: List[str] = []
            summary_lines: List[str] = []
            capturing_bullets = False

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.lower().startswith("key takeaway") or stripped.lower().startswith("key point"):
                    capturing_bullets = True
                    continue
                if stripped.startswith("- ") or stripped.startswith("* ") or re.match(r"^\d+\.\s+", stripped):
                    clean_bullet = re.sub(r"^[-*\d.]+\s*", "", stripped)
                    key_points.append(clean_bullet)
                elif not capturing_bullets:
                    summary_lines.append(stripped)

            summary_text = "\n\n".join(summary_lines) if summary_lines else raw_response

            return DocumentSummaryResponse(
                document_id=doc.id,
                file_name=doc.file_name,
                document_type=doc.document_type,
                summary=summary_text,
                key_points=key_points,
                model_id=model_id,
            )

        except Exception as e:
            logger.error("Failed to generate summary for document ID %d: %s", document_id, str(e))
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document summarization failed: {str(e)}",
            ) from e

    def extract_structured_info(self, document_id: int) -> DocumentExtractionResponse:
        """Extract structured insurance data (policy type, coverage, exclusions, deductible,
        important conditions, risk indicators) into validated JSON without hallucination.

        Args:
            document_id: Database primary key of the document.

        Returns:
            DocumentExtractionResponse: Validated structured insurance extraction.
        """
        doc, full_text = self._assemble_document_text(document_id)
        logger.info("Extracting structured info for document ID %d ('%s')...", document_id, doc.file_name)

        system_prompt = (
            "You are a specialized insurance data extraction engine. "
            "Extract structured insurance attributes from the provided document. "
            "You must strictly output valid JSON only, without commentary or markdown codeblocks. "
            "DO NOT INVENT, ASSUME, OR GUESS facts not explicitly stated in the source text. "
            "If any field or item is not mentioned in the source document, set it to 'Not Specified' or an empty list."
        )

        user_prompt = (
            f"Analyze the insurance document text below and extract structured information into JSON.\n\n"
            f"Document Name: {doc.file_name}\n"
            f"Document Category: {doc.document_type}\n\n"
            f"Document Content:\n{full_text}\n\n"
            f"JSON Output Structure:\n"
            f"{{\n"
            f'  "policy_type": "string describing type of policy or Not Specified",\n'
            f'  "coverage": "detailed dictionary, list, or summary of monetary limits and covered property/liabilities",\n'
            f'  "exclusions": ["list of explicit perils or situations excluded"],\n'
            f'  "deductible": "monetary or percentage deductible terms stated in the text",\n'
            f'  "important_conditions": ["list of policyholder requirements, notice timelines, warranties"],\n'
            f'  "risk_indicators": ["list of observations regarding high deductibles, notable exclusions, or exposure flags"]\n'
            f"}}\n\n"
            f"Ensure the response is valid parseable JSON without any extra introductory or concluding text."
        )

        try:
            model_id = getattr(self._llm_service, "model_id", "amazon.nova-lite-v1:0")
            raw_response = self._llm_service.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,  # Strict deterministic extraction
            )

            # Clean JSON markdown fences if returned
            clean_json_str = raw_response.strip()
            if clean_json_str.startswith("```json"):
                clean_json_str = clean_json_str[len("```json"):].strip()
            elif clean_json_str.startswith("```"):
                clean_json_str = clean_json_str[len("```"):].strip()
            if clean_json_str.endswith("```"):
                clean_json_str = clean_json_str[:-3].strip()

            # Find outer JSON braces
            first_brace = clean_json_str.find("{")
            last_brace = clean_json_str.rfind("}")
            if first_brace != -1 and last_brace != -1:
                clean_json_str = clean_json_str[first_brace : last_brace + 1]

            try:
                parsed_data = json.loads(clean_json_str)
                extraction = StructuredInsuranceExtraction(**parsed_data)
            except (json.JSONDecodeError, ValueError) as parse_err:
                logger.warning("Could not parse JSON cleanly: %s. Using partial structure.", parse_err)
                extraction = StructuredInsuranceExtraction(
                    policy_type=doc.document_type,
                    coverage="See raw response",
                    exclusions=[],
                    deductible="See raw response",
                    important_conditions=[],
                    risk_indicators=[],
                )

            return DocumentExtractionResponse(
                document_id=doc.id,
                file_name=doc.file_name,
                document_type=doc.document_type,
                model_id=model_id,
                extraction=extraction,
                raw_response=raw_response,
            )

        except Exception as e:
            logger.error("Failed to extract structured data for document ID %d: %s", document_id, str(e))
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Structured extraction failed: {str(e)}",
            ) from e
