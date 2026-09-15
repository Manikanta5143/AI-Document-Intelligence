from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger("text_chunker")


class TextChunker:
    """Splits document text into overlapping segments with configurable chunk size and overlap."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Ensure valid parameters
        if self.chunk_size <= 0:
            self.chunk_size = 1000
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 5)

        self.step_size = self.chunk_size - self.chunk_overlap

    def _split_page_text(self, text: str) -> List[str]:
        """Split a single block of text into overlapping character slices.

        Attempts to break at space or newline boundaries when available.
        """
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        total_len = len(text)

        while start < total_len:
            end = start + self.chunk_size

            # If this is not the last chunk, try to find a natural boundary (newline or space)
            if end < total_len:
                boundary = -1
                search_start = max(start + self.chunk_overlap, end - 100)
                # Look for paragraph or line break first
                newline_pos = text.rfind("\n", search_start, end)
                if newline_pos != -1:
                    boundary = newline_pos + 1
                else:
                    # Look for whitespace
                    space_pos = text.rfind(" ", search_start, end)
                    if space_pos != -1:
                        boundary = space_pos + 1

                if boundary > start:
                    end = boundary

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= total_len:
                break

            start += self.step_size

        return chunks

    def chunk_document_pages(
        self,
        pages_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Chunk a list of page objects while tracking page numbers and sequential index.

        Args:
            pages_data: List of dicts with 'page_number' and 'text'.

        Returns:
            List[Dict[str, Any]]: List of chunk dicts containing:
                - chunk_index: int (0-based sequential counter)
                - page_number: int (1-based page where text originated)
                - chunk_text: str (content of chunk)
        """
        chunks_result: List[Dict[str, Any]] = []
        chunk_counter = 0

        for page in pages_data:
            page_number = page.get("page_number", 1)
            page_text = page.get("text", "")

            if not page_text.strip():
                continue

            page_chunks = self._split_page_text(page_text)
            for text_segment in page_chunks:
                chunks_result.append({
                    "chunk_index": chunk_counter,
                    "page_number": page_number,
                    "chunk_text": text_segment,
                })
                chunk_counter += 1

        logger.info(
            "Created %d chunks across %d pages (size=%d, overlap=%d).",
            len(chunks_result),
            len(pages_data),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks_result
