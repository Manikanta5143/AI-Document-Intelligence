import json
from app.rag.vector_store import get_vector_store_service

with open('evaluation_results/question_ground_truth.json') as f:
    gt = json.load(f)

vs = get_vector_store_service()
print("Testing retrieval for all 15 questions (top_k=5)...")
for q in gt["questions"]:
    chunks = vs.similarity_search(query=q["question"], top_k=5, document_id=3)
    retrieved_ids = [c["chunk_id"] for c in chunks]
    print(f"{q['question_id']}: expected={q['relevant_chunk_ids']} -> retrieved={retrieved_ids}")
