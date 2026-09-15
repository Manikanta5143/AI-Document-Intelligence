from fastapi import APIRouter, Depends, HTTPException, status

from app.rag.vector_store import VectorStoreService, get_vector_store_service
from app.schemas.search import SearchRequest, SearchResponse, SearchResultItem

router = APIRouter(prefix="/search", tags=["Search & Retrieval"])


@router.post(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Similarity Search",
    description=(
        "Performs semantic similarity search against indexed insurance document chunks in ChromaDB. "
        "Generates a query embedding using Amazon Bedrock Titan Text Embeddings, retrieves the top-k "
        "most relevant chunks, and supports optional metadata filtering by document_id and document_type."
    ),
)
def semantic_search(
    request: SearchRequest,
    vector_store: VectorStoreService = Depends(get_vector_store_service),
) -> SearchResponse:
    """Execute semantic similarity search against ChromaDB."""
    try:
        results = vector_store.similarity_search(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
            document_type=request.document_type,
        )
        items = [SearchResultItem(**item) for item in results]
        return SearchResponse(
            query=request.query,
            total_results=len(items),
            results=items,
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        ) from ve
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}",
        ) from e
