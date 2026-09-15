import re
from typing import Any, Dict, List
# pyrefly: ignore [missing-import]
import pymupdf

from app.utils.logger import setup_logger

logger = setup_logger("pdf_extractor")


class PDFExtractor:
    """Extracts and cleans text from PDF documents page-by-page using PyMuPDF."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Sanitize and normalize extracted PDF text.

        Removes null bytes, non-printable control characters, and excess whitespace
        while preserving paragraph structure.

        Args:
            text: Raw string extracted from PDF page.

        Returns:
            str: Normalized, readable text.
        """
        if not text:
            return ""

        # Remove null characters and non-printable control chars (preserve \n, \t)
        text = text.replace("\x00", "")
        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize redundant spaces and tabs within each line
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

        # Recombine lines, collapsing 3+ consecutive newlines to 2 (paragraph break)
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    def extract_pages(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract text from all pages of a PDF document.

        Args:
            pdf_bytes: Raw binary bytes of the PDF file.

        Raises:
            ValueError: If the byte stream is not a valid PDF or has no pages.

        Returns:
            List[Dict[str, Any]]: List containing dicts with:
                - page_number: int (1-indexed)
                - text: str (cleaned text)
                - char_count: int
        """
        if not pdf_bytes:
            raise ValueError("Cannot extract text from empty document bytes.")

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            logger.error("PyMuPDF failed to open PDF document: %s", str(e))
            raise ValueError(f"Invalid or corrupted PDF file: {str(e)}") from e

        total_pages = doc.page_count
        if total_pages == 0:
            doc.close()
            raise ValueError("The provided PDF has 0 pages.")

        logger.info("Extracting text from PDF (%d pages)...", total_pages)
        pages_data = []

        try:
            for page_idx in range(total_pages):
                page = doc.load_page(page_idx)
                raw_text = page.get_text("text") or ""
                cleaned = self.clean_text(raw_text)
                page_num = page_idx + 1

                pages_data.append({
                    "page_number": page_num,
                    "text": cleaned,
                    "char_count": len(cleaned),
                })
        finally:
            doc.close()

        logger.info("Extracted %d pages successfully.", len(pages_data))
        return pages_data
