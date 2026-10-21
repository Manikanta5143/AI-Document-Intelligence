"""RAG Evaluation Engine and Metric Calculator for Insurance AI Document Intelligence.

Calculates retrieval metrics (Precision@k, Recall@k, F1@k, MRR@k, nDCG@k)
and generation metrics (Answer Correctness, Completeness, Fact Precision/Recall/F1,
Faithfulness, Groundedness, Hallucination Rate, Citation Quality, Abstention Accuracy).
"""
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple


def calculate_retrieval_metrics(
    retrieved_chunk_ids: List[int],
    ground_truth_relevant_ids: List[int],
    k: int,
    latency_ms: float = 0.0,
) -> Dict[str, Any]:
    """Calculate standard information retrieval metrics at rank k.

    Args:
        retrieved_chunk_ids: Ranked list of retrieved chunk IDs.
        ground_truth_relevant_ids: Verified list of relevant chunk IDs.
        k: Cutoff rank (top_k).
        latency_ms: Retrieval latency in milliseconds.

    Returns:
        Dict containing precision, recall, f1, mrr, ndcg, and coverage.
    """
    cutoff_retrieved = retrieved_chunk_ids[:k]
    gt_set = set(ground_truth_relevant_ids)

    # For unanswerable questions where no relevant chunks exist in the document:
    if len(gt_set) == 0:
        return {
            "retrieval_precision": 0.0,
            "retrieval_recall": None,
            "retrieval_f1": None,
            "mrr": None,
            "ndcg": None,
            "context_relevance": 0.0,
            "context_coverage": None,
            "num_retrieved_chunks": len(cutoff_retrieved),
            "retrieval_latency_ms": latency_ms,
        }

    relevant_retrieved = [cid for cid in cutoff_retrieved if cid in gt_set]
    num_relevant_retrieved = len(relevant_retrieved)

    # Precision@k = |Retrieved & Relevant| / |Retrieved|
    precision = num_relevant_retrieved / k if k > 0 else 0.0

    # Recall@k = |Retrieved & Relevant| / |Relevant|
    recall = num_relevant_retrieved / len(gt_set) if len(gt_set) > 0 else 0.0

    # F1@k
    if (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # MRR@k (Mean Reciprocal Rank of first relevant chunk)
    mrr = 0.0
    for rank_idx, cid in enumerate(cutoff_retrieved, start=1):
        if cid in gt_set:
            mrr = 1.0 / rank_idx
            break

    # nDCG@k with binary relevance
    dcg = 0.0
    for rank_idx, cid in enumerate(cutoff_retrieved, start=1):
        rel = 1.0 if cid in gt_set else 0.0
        dcg += rel / math.log2(rank_idx + 1)

    # Ideal DCG@k
    ideal_relevant_count = min(k, len(gt_set))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_relevant_count + 1))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "retrieval_precision": round(precision, 4),
        "retrieval_recall": round(recall, 4),
        "retrieval_f1": round(f1, 4),
        "mrr": round(mrr, 4),
        "ndcg": round(ndcg, 4),
        "context_relevance": round(precision, 4),
        "context_coverage": round(recall, 4),
        "num_retrieved_chunks": len(cutoff_retrieved),
        "retrieval_latency_ms": round(latency_ms, 2),
    }


def evaluate_abstention(generated_answer: str) -> Tuple[bool, str]:
    """Check whether model properly abstains when information is missing."""
    ans_lower = generated_answer.lower()
    abstain_patterns = [
        "does not provide",
        "does not contain",
        "does not mention",
        "does not state",
        "does not include",
        "not mentioned",
        "not provided",
        "not specified",
        "not available",
        "no information",
        "cannot be determined",
        "unable to determine",
        "not found in the document",
        "not included in the provided",
        "not present in the",
        "insufficient information",
    ]
    for p in abstain_patterns:
        if p in ans_lower:
            return True, p
    return False, ""


def evaluate_citations(
    generated_answer: str,
    retrieved_chunk_ids: List[int],
    ground_truth_relevant_ids: List[int],
) -> Dict[str, Any]:
    """Assess whether model cites pages/excerpts accurately and with high coverage."""
    ans = generated_answer
    has_page_citation = bool(re.search(r"\bpage\s*\d+\b", ans, re.IGNORECASE))
    has_excerpt_citation = bool(re.search(r"\bexcerpt\s*\d+\b|\bchunk\s*\d*\b", ans, re.IGNORECASE))
    has_source_header = bool(re.search(r"\bsource\b|\breferences?\b", ans, re.IGNORECASE))

    has_any_citation = has_page_citation or has_excerpt_citation or has_source_header

    # All resume chunks are Page 1
    # Check if any cited page is invalid (e.g. Page 2, Page 3)
    page_numbers = [int(m) for m in re.findall(r"\bpage\s*(\d+)\b", ans, re.IGNORECASE)]
    invalid_pages = [p for p in page_numbers if p != 1]

    citation_precision = 1.0 if (not invalid_pages and has_any_citation) else (0.5 if invalid_pages else 0.0)
    citation_coverage = 1.0 if has_any_citation else 0.0

    return {
        "citation_precision": round(citation_precision, 4),
        "citation_coverage": round(citation_coverage, 4),
        "has_page_citation": has_page_citation,
        "has_excerpt_citation": has_excerpt_citation,
        "cited_pages": page_numbers,
    }


def evaluate_question_answer(
    question_info: Dict[str, Any],
    generated_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    temperature: float,
    top_k: int,
) -> Dict[str, Any]:
    """Evaluate generated answer against expected facts, retrieved context, and reference answer."""
    qid = question_info["question_id"]
    is_answerable = question_info["is_answerable"]
    expected_facts = question_info["expected_facts"]
    relevant_chunk_ids = set(question_info["relevant_chunk_ids"])
    retrieved_chunk_ids = [c["chunk_id"] for c in retrieved_chunks[:top_k]]
    retrieved_context_text = "\n".join([c["text"] for c in retrieved_chunks[:top_k]])

    ans = generated_answer.strip()
    ans_lower = ans.lower()

    supported_facts: List[str] = []
    missing_facts: List[str] = []
    unsupported_claims: List[str] = []
    contradictions: List[str] = []

    # 1. Category E: Unanswerable Questions
    if not is_answerable:
        abstained, pattern = evaluate_abstention(ans)
        abstention_correct = abstained

        # Check for hallucinations on unanswerable questions
        hallucinated = False
        if qid == "Q13":
            # Check if invented any percentage improvement
            pct_matches = re.findall(r"\b\d+(?:\.\d+)?%", ans)
            if pct_matches:
                hallucinated = True
                unsupported_claims.append(f"Invented accuracy percentage: {pct_matches}")
        elif qid == "Q14":
            # Check if invented ms latency or dollar cost
            ms_matches = re.findall(r"\b\d+\s*ms\b", ans_lower)
            dollar_matches = re.findall(r"\$\s*\d+", ans)
            if ms_matches or dollar_matches:
                hallucinated = True
                unsupported_claims.append(f"Invented latency or cost: {ms_matches + dollar_matches}")
        elif qid == "Q15":
            # Check if claimed Stampede Recognition exists or gave F1 score
            f1_matches = re.findall(r"\b(?:f1|precision)\b[^\n.]{0,30}\b\d+(?:\.\d+)?", ans_lower)
            if ("stampede recognition" in ans_lower and "not mention" not in ans_lower and "does not" not in ans_lower) or f1_matches:
                hallucinated = True
                unsupported_claims.append(f"Invented Stampede Recognition model metrics: {f1_matches}")

        if abstention_correct and not hallucinated:
            supported_facts.append("Correctly abstained due to missing information in source document.")
            faithfulness_score = 1.0
            groundedness_score = 1.0
            hallucination_rate = 0.0
            answer_correctness_score = 1.0
            completeness_score = 1.0
            answer_precision = 1.0
            answer_recall = 1.0
            answer_f1 = 1.0
        else:
            missing_facts.append("Failed to abstain; asserted unsupported claims.")
            faithfulness_score = 0.0
            groundedness_score = 0.0
            hallucination_rate = 1.0
            answer_correctness_score = 0.0
            completeness_score = 0.0
            answer_precision = 0.0
            answer_recall = 0.0
            answer_f1 = 0.0

        cit = evaluate_citations(ans, retrieved_chunk_ids, list(relevant_chunk_ids))
        return {
            "expected_facts": expected_facts,
            "supported_facts": supported_facts,
            "missing_facts": missing_facts,
            "unsupported_claims": unsupported_claims,
            "contradictions": contradictions,
            "answer_precision": answer_precision,
            "answer_recall": answer_recall,
            "answer_f1": answer_f1,
            "faithfulness_score": faithfulness_score,
            "groundedness_score": groundedness_score,
            "answer_correctness_score": answer_correctness_score,
            "completeness_score": completeness_score,
            "hallucination_rate": hallucination_rate,
            "abstention_correct": abstention_correct,
            "citation_precision": cit["citation_precision"],
            "citation_coverage": cit["citation_coverage"],
            "evaluation_notes": f"Unanswerable question. Abstention: {abstention_correct} (pattern: '{pattern}'). Hallucinations: {len(unsupported_claims)}.",
        }

    # 2. Answerable Questions (Categories A, B, C, D)
    # Check if model falsely abstained
    abstained, pattern = evaluate_abstention(ans)
    # If the model abstained when relevant context was present in the chunks, that's a false abstention
    relevant_chunks_in_context = any(cid in relevant_chunk_ids for cid in retrieved_chunk_ids)

    # Detailed fact verification rules per question
    fact_evaluators = {
        "Q01": [
            ("Titan Embeddings V2", lambda a: "titan" in a and ("embed" in a or "v2" in a)),
        ],
        "Q02": [
            ("CGPA 8.88/10", lambda a: "8.88" in a),
            ("B.Tech CSE (AI & ML) JNTUH Class of 2027", lambda a: ("computer science" in a or "cse" in a or "ai & ml" in a) and ("jawaharlal nehru" in a or "jntu" in a or "2027" in a)),
        ],
        "Q03": [
            ("ChromaDB vector store", lambda a: "chromadb" in a or "chroma" in a),
            ("FastAPI backend REST API", lambda a: "fastapi" in a),
        ],
        "Q04": [
            ("Programming languages Python, Java, C++", lambda a: "python" in a and "java" in a and "c++" in a),
            ("Design patterns Singleton, Factory, Builder, Strategy, Observer", lambda a: sum(p in a for p in ["singleton", "factory", "builder", "strategy", "observer"]) >= 3),
        ],
        "Q05": [
            ("Storage in Amazon S3", lambda a: "s3" in a or "amazon s3" in a),
            ("PyMuPDF PDF extraction", lambda a: "pymupdf" in a),
            ("Titan Text Embeddings V2", lambda a: "titan" in a),
            ("ChromaDB vector similarity search", lambda a: "chromadb" in a or "chroma" in a),
            ("Amazon Bedrock Nova 2 Lite generation", lambda a: "nova" in a or "bedrock" in a),
            ("MySQL for metadata/Q&A, Snowflake for analytics", lambda a: "mysql" in a and "snowflake" in a),
        ],
        "Q06": [
            ("Amazon S3 stores policy documents", lambda a: "s3" in a and ("document" in a or "store" in a or "policy" in a)),
            ("MySQL stores document metadata and Q&A history", lambda a: "mysql" in a and ("metadata" in a or "history" in a or "query" in a or "q&a" in a)),
            ("Snowflake stores and analyzes application analytics", lambda a: "snowflake" in a and "analytics" in a),
        ],
        "Q07": [
            ("Titan Text Embeddings V2 generates vector embeddings", lambda a: "titan" in a and "embed" in a),
            ("ChromaDB indexes and performs similarity search", lambda a: ("chromadb" in a or "chroma" in a) and ("store" in a or "search" in a or "similarity" in a or "retriev" in a)),
            ("Bedrock Nova 2 Lite generates context-grounded answers", lambda a: ("nova" in a or "bedrock" in a) and ("answer" in a or "generate" in a or "ground" in a)),
        ],
        "Q08": [
            ("Zcoder tech stack (React, Node, Express, MongoDB, Socket.IO)", lambda a: "react" in a and "node" in a and "express" in a and "mongodb" in a),
            ("Stateless JWT, bcrypt, Nodemailer with UUID tokens", lambda a: "jwt" in a and "bcrypt" in a and ("nodemailer" in a or "uuid" in a or "token" in a)),
            ("Multi-language code execution for 100+ problems", lambda a: "code execution" in a or "100+" in a or "problem" in a),
            ("Modular REST APIs and Mongoose models for problems/contests/profiles", lambda a: "rest" in a or "mongoose" in a or "contest" in a or "profile" in a),
            ("Real-time chat with Socket.IO bidirectional communication", lambda a: "socket.io" in a or "chat" in a),
        ],
        "Q09": [
            ("Responsive React.js frontend with React Router/Axios", lambda a: "react" in a and ("router" in a or "axios" in a or "frontend" in a)),
            ("Node.js/Express.js backend RESTful APIs", lambda a: ("node" in a or "express" in a) and "api" in a),
            ("JWT authentication and bcrypt hashing", lambda a: "jwt" in a and "bcrypt" in a),
            ("MongoDB database", lambda a: "mongodb" in a or "mongo" in a),
            ("Aggregates 50+ hackathons/internships with search, filter, bookmarking", lambda a: ("50+" in a or "hackathon" in a or "internship" in a) and ("search" in a or "filter" in a or "bookmark" in a)),
        ],
        "Q10": [
            ("Education: B.Tech CSE (AI & ML) JNTUH CGPA 8.88/10", lambda a: "8.88" in a and ("jntuh" in a or "jawaharlal" in a or "b.tech" in a or "class of 2027" in a)),
            ("Technical skills: Python/Java/C++, FastAPI, Node, React, Bedrock, databases", lambda a: ("python" in a or "fastapi" in a) and ("react" in a or "node" in a)),
            ("Insurance AI Assistant RAG project", lambda a: "insurance" in a and ("rag" in a or "bedrock" in a or "chromadb" in a)),
            ("Zcoder coding platform project", lambda a: "zcoder" in a),
            ("Dolphin opportunity portal project", lambda a: "dolphin" in a),
            ("Achievements: 600+ DSA (LeetCode/GFG) & Walmart Hackathon", lambda a: "600+" in a or "dsa" in a or "walmart" in a or "leetcode" in a),
        ],
        "Q11": [
            ("Insurance AI Assistant: AI/ML & Backend (FastAPI, Bedrock, Titan, ChromaDB, S3)", lambda a: "insurance" in a and ("fastapi" in a or "bedrock" in a or "titan" in a or "chromadb" in a)),
            ("Zcoder: Real-time & Backend (Socket.IO, Node, Express, MongoDB, JWT)", lambda a: "zcoder" in a and ("socket.io" in a or "chat" in a or "real-time" in a)),
            ("Dolphin: Backend development (Node, Express, MongoDB, JWT)", lambda a: "dolphin" in a and ("node" in a or "express" in a or "mongodb" in a or "jwt" in a)),
        ],
        "Q12": [
            ("Stateless JWT authentication in Zcoder and Dolphin", lambda a: "jwt" in a and ("zcoder" in a or "dolphin" in a or "stateless" in a)),
            ("bcrypt password hashing in Zcoder and Dolphin", lambda a: "bcrypt" in a and ("password" in a or "hash" in a)),
            ("Nodemailer email verification with UUID tokens in Zcoder", lambda a: ("nodemailer" in a or "uuid" in a) and ("token" in a or "email" in a or "verification" in a)),
        ],
    }

    eval_rules = fact_evaluators.get(qid, [])
    for fact_desc, test_fn in eval_rules:
        if test_fn(ans_lower):
            supported_facts.append(fact_desc)
        else:
            missing_facts.append(fact_desc)

    total_expected = len(eval_rules) if eval_rules else len(expected_facts)
    num_supported = len(supported_facts)

    # Answer Recall = supported expected facts / total expected facts
    answer_recall = (num_supported / total_expected) if total_expected > 0 else 0.0

    # Extract claims made:
    # Any claim in answer that is NOT in retrieved context or contradicts it is unsupported
    # Check groundedness in retrieved context:
    # Check whether the facts stated by the model actually exist in retrieved chunks!
    context_lower = retrieved_context_text.lower()
    for fact_desc, test_fn in eval_rules:
        if test_fn(ans_lower):
            # If the model asserted this fact, did it actually exist in the retrieved context?
            if not test_fn(context_lower):
                # The model asserted a fact that was NOT in the retrieved context!
                # (e.g. at low top_k where chunk 11 was omitted, but model recited it from prior knowledge or pretraining)
                unsupported_claims.append(f"Fact asserted without context support: {fact_desc}")

    # Check for contradictions
    if "8.88" in ans and ("7." in ans or "9." in ans):
        contradictions.append("Conflicting CGPA numbers")
    if "pgvector" in ans_lower or "pinecone" in ans_lower or "qdrant" in ans_lower:
        contradictions.append("Hallucinated alternative vector database (not ChromaDB)")

    # Answer precision: supported claims / (supported claims + unsupported claims + contradictions)
    total_claims = num_supported + len(unsupported_claims) + len(contradictions)
    answer_precision = (num_supported / total_claims) if total_claims > 0 else 0.0

    # Answer F1
    if (answer_precision + answer_recall) > 0:
        answer_f1 = 2.0 * answer_precision * answer_recall / (answer_precision + answer_recall)
    else:
        answer_f1 = 0.0

    # Faithfulness: supported by retrieved context / total claims made
    context_supported_count = num_supported - len(unsupported_claims)
    context_supported_count = max(0, context_supported_count)
    faithfulness_score = (context_supported_count / total_claims) if total_claims > 0 else 1.0

    # Groundedness score
    groundedness_score = faithfulness_score

    # Hallucination rate = unsupported or contradicted claims / total claims
    hallucination_rate = ((len(unsupported_claims) + len(contradictions)) / total_claims) if total_claims > 0 else 0.0

    # Completeness score
    completeness_score = answer_recall

    # Correctness score: penalized if missing facts or unsupported claims
    answer_correctness_score = max(0.0, answer_recall - (0.5 * hallucination_rate))

    # Abstention correct: True if it didn't falsely abstain when relevant context was present
    abstention_correct = not (abstained and relevant_chunks_in_context)

    cit = evaluate_citations(ans, retrieved_chunk_ids, list(relevant_chunk_ids))

    return {
        "expected_facts": [f[0] for f in eval_rules] if eval_rules else expected_facts,
        "supported_facts": supported_facts,
        "missing_facts": missing_facts,
        "unsupported_claims": unsupported_claims,
        "contradictions": contradictions,
        "answer_precision": round(answer_precision, 4),
        "answer_recall": round(answer_recall, 4),
        "answer_f1": round(answer_f1, 4),
        "faithfulness_score": round(faithfulness_score, 4),
        "groundedness_score": round(groundedness_score, 4),
        "answer_correctness_score": round(answer_correctness_score, 4),
        "completeness_score": round(completeness_score, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "abstention_correct": abstention_correct,
        "citation_precision": cit["citation_precision"],
        "citation_coverage": cit["citation_coverage"],
        "evaluation_notes": (
            f"Recall: {num_supported}/{total_expected}. "
            f"Unsupported claims: {len(unsupported_claims)}. "
            f"Contradictions: {len(contradictions)}."
        ),
    }
