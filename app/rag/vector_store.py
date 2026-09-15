import os
from typing import Any, Dict, List, Optional
import chromadb

from app.aws.bedrock_service import BaseEmbeddingService, get_embedding_service
from app.config import Settings, get_settings
from app.models.chunk import DocumentChunk
from app.utils.logger import setup_logger

logger = setup_logger("vector_store")


class VectorStoreService:
    """Manages persistent ChromaDB vector collection and semantic similarity search."""

    def __init__(
        self,
        embedding_service: Optional[BaseEmbeddingService] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or get_embedding_service()
        self.persist_directory = os.path.abspath(self.settings.chroma_persist_directory)
        self.collection_name = self.settings.chroma_collection_name

        os.makedirs(self.persist_directory, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_directory)

        # Initialize or fetch persistent collection using cosine distance metric
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Connected to ChromaDB (dir='%s', collection='%s', items=%d)",
            self.persist_directory,
            self.collection_name,
            self._collection.count(),
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Embed and index document chunks into ChromaDB with comprehensive metadata.

        Preserves:
            - embedding vector (via Bedrock or injected service)
            - chunk text (document)
            - document_id
            - document_name (file_name)
            - page_number
            - chunk_id
            - document_type

        Args:
            chunks: List of DocumentChunk model instances.

        Returns:
            int: Number of chunks successfully indexed.
        """
        if not chunks:
            return 0

        logger.info("Indexing %d chunks into ChromaDB collection '%s'...", len(chunks), self.collection_name)

        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            chunk_id_str = f"doc_{chunk.document_id}_chunk_{chunk.id or chunk.chunk_index}"
            ids.append(chunk_id_str)
            texts.append(chunk.chunk_text)
            metadatas.append({
                "document_id": int(chunk.document_id),
                "document_name": str(chunk.file_name or "unknown"),
                "document_type": str(chunk.document_type or "Unknown"),
                "page_number": int(chunk.page_number),
                "chunk_id": int(chunk.id or chunk.chunk_index),
                "chunk_index": int(chunk.chunk_index),
            })

        # 1. Generate embeddings
        embeddings = self.embedding_service.embed_documents(texts)

        # 2. Persist in ChromaDB collection
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(
            "Successfully indexed %d chunks in ChromaDB. Total collection size: %d",
            len(chunks),
            self._collection.count(),
        )
        return len(chunks)

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[int] = None,
        document_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k semantically relevant chunks for a question.

        Flow:
            1. Generate query embedding vector.
            2. Build metadata filter (document_id, document_type) if requested.
            3. Query ChromaDB collection with cosine similarity.
            4. Format and return top-k chunks with scores.

        Args:
            query: Question or search phrase.
            top_k: Number of chunks to retrieve (default 5).
            document_id: Optional document ID filter.
            document_type: Optional document type filter.

        Returns:
            List[Dict[str, Any]]: Ranked chunks with similarity score and metadata.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        clean_query = query.strip()
        logger.info("Executing semantic search for '%s' (top_k=%d)...", clean_query, top_k)

        # 1. Generate query embedding
        query_embedding = self.embedding_service.embed_text(clean_query)

        # 2. Build metadata filter
        filters = []
        if document_id is not None:
            filters.append({"document_id": int(document_id)})
        if document_type is not None:
            filters.append({"document_type": str(document_type)})

        where_filter = None
        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}

        # 3. Query ChromaDB
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, max(1, self._collection.count() or 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self._collection.query(**query_kwargs)

        parsed_results: List[Dict[str, Any]] = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            doc_ids = results["ids"][0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(doc_ids)):
                dist = distances[i] if i < len(distances) else 0.0
                # ChromaDB cosine distance ranges from 0.0 (identical) upwards
                similarity = max(0.0, min(1.0, 1.0 - dist))
                meta = metadatas[i] if i < len(metadatas) else {}

                parsed_results.append({
                    "chunk_id": meta.get("chunk_id", 0),
                    "document_id": meta.get("document_id", 0),
                    "document_name": meta.get("document_name", ""),
                    "document_type": meta.get("document_type", ""),
                    "page_number": meta.get("page_number", 1),
                    "text": documents[i] if i < len(documents) else "",
                    "similarity_score": round(similarity, 4),
                })

        logger.info("Retrieved %d relevant chunks for query.", len(parsed_results))
        return parsed_results

    def get_collection_count(self) -> int:
        """Return total number of indexed chunk vectors in ChromaDB."""
        return self._collection.count()


def get_vector_store_service() -> VectorStoreService:
    """Dependency provider for VectorStoreService."""
    return VectorStoreService()
