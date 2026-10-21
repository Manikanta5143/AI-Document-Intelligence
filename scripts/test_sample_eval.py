import json
from app.aws.llm_service import BedrockLLMService
from app.rag.vector_store import get_vector_store_service
from app.database.session import get_db
from app.repositories.document_repository import DocumentRepository

db = next(get_db())
doc_repo = DocumentRepository(db)
doc = doc_repo.get_by_id(3)
vs = get_vector_store_service()
llm = BedrockLLMService()

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

for q_text in [
    "Which embedding model does my Insurance AI Assistant use for document and query embeddings?",
    "What exact percentage improvement in document retrieval accuracy did my Insurance AI Assistant achieve compared to standard keyword search?"
]:
    chunks = vs.similarity_search(query=q_text, top_k=2, document_id=3)
    context_blocks = [f"[Excerpt {i+1} | Page {c['page_number']} | Chunk ID: {c['chunk_id']}]\n{c['text']}" for i, c in enumerate(chunks)]
    retrieved_context = "\n\n".join(context_blocks)
    prompt = (
        f"Document Name: {doc.file_name}\n"
        f"Document Category: {doc.document_type}\n\n"
        f"Retrieved Document Context:\n"
        f"{retrieved_context}\n\n"
        f"User Question: {q_text}\n\n"
        f"Answer:"
    )
    meta = llm.generate_text_with_metadata(prompt=prompt, system_prompt=system_instruction, temperature=0.0)
    print(f"=== Question: {q_text} ===")
    print(f"Answer: {meta['text']}")
    print(f"Tokens: in={meta['input_tokens']}, out={meta['output_tokens']}, latency={meta['latency_ms']}ms")
    print()
