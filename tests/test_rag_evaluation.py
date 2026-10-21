"""Automated tests for parameter propagation and metric calculation in Insurance AI RAG pipeline.

Compatible with standard library unittest and pytest.
"""
import math
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pydantic import ValidationError

from app.schemas.rag import RAGQuestionRequest
from scripts.rag_eval_engine import (
    calculate_retrieval_metrics,
    evaluate_abstention,
    evaluate_citations,
    evaluate_question_answer,
)


class TestParameterPropagation(unittest.TestCase):
    """Tests schema validation and parameter bounds for top_k and temperature."""

    def test_valid_parameters(self):
        req = RAGQuestionRequest(
            question="What is the policy deductible?",
            top_k=3,
            temperature=0.2,
        )
        self.assertEqual(req.top_k, 3)
        self.assertEqual(req.temperature, 0.2)

    def test_default_optional_parameters(self):
        req = RAGQuestionRequest(question="What is the policy deductible?")
        self.assertIsNone(req.top_k)
        self.assertIsNone(req.temperature)

    def test_top_k_boundaries(self):
        # top_k must be between 1 and 20
        with self.assertRaises(ValidationError):
            RAGQuestionRequest(question="Test?", top_k=0)

        with self.assertRaises(ValidationError):
            RAGQuestionRequest(question="Test?", top_k=21)

        req_min = RAGQuestionRequest(question="Test?", top_k=1)
        self.assertEqual(req_min.top_k, 1)

        req_max = RAGQuestionRequest(question="Test?", top_k=20)
        self.assertEqual(req_max.top_k, 20)

    def test_temperature_boundaries(self):
        # temperature must be between 0.0 and 1.0
        with self.assertRaises(ValidationError):
            RAGQuestionRequest(question="Test?", temperature=-0.1)

        with self.assertRaises(ValidationError):
            RAGQuestionRequest(question="Test?", temperature=1.1)

        req_zero = RAGQuestionRequest(question="Test?", temperature=0.0)
        self.assertEqual(req_zero.temperature, 0.0)

        req_one = RAGQuestionRequest(question="Test?", temperature=1.0)
        self.assertEqual(req_one.temperature, 1.0)

    def test_rag_service_parameter_propagation(self):
        from unittest.mock import MagicMock
        from app.rag.rag_service import RAGService
        from app.models.document import Document

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            {"chunk_id": 1, "page_number": 1, "text": "Test chunk"}
        ]
        mock_llm = MagicMock()
        mock_llm.generate_text.return_value = "Test response"
        mock_doc_repo = MagicMock()
        mock_doc = Document(id=1, file_name="policy.pdf", document_type="Policy", status="PROCESSED")
        mock_doc_repo.get_by_id.return_value = mock_doc
        mock_query_repo = MagicMock()
        mock_query_repo.create.return_value = MagicMock(id=101)

        rag = RAGService(
            vector_store=mock_vs,
            llm_service=mock_llm,
            document_repository=mock_doc_repo,
            query_repository=mock_query_repo,
        )

        resp = rag.ask_question(document_id=1, question="What is deductible?", top_k=4, temperature=0.3)

        # Verify top_k was passed to vector store similarity search
        mock_vs.similarity_search.assert_called_once_with(
            query="What is deductible?",
            top_k=4,
            document_id=1,
        )

        # Verify temperature was passed to LLM generate_text
        _, kwargs = mock_llm.generate_text.call_args
        self.assertEqual(kwargs.get("temperature"), 0.3)


class TestMetricCalculations(unittest.TestCase):
    """Tests retrieval and generation metric calculation functions."""

    def test_perfect_retrieval_metrics(self):
        retrieved = [12, 13]
        relevant = [12, 13]
        m = calculate_retrieval_metrics(retrieved, relevant, k=2, latency_ms=10.0)

        self.assertEqual(m["retrieval_precision"], 1.0)
        self.assertEqual(m["retrieval_recall"], 1.0)
        self.assertEqual(m["retrieval_f1"], 1.0)
        self.assertEqual(m["mrr"], 1.0)
        self.assertEqual(m["ndcg"], 1.0)
        self.assertEqual(m["num_retrieved_chunks"], 2)

    def test_partial_retrieval_metrics(self):
        # retrieved: [12, 15], but only 12 is relevant out of [12, 13]
        retrieved = [12, 15]
        relevant = [12, 13]
        m = calculate_retrieval_metrics(retrieved, relevant, k=2, latency_ms=15.0)

        self.assertEqual(m["retrieval_precision"], 0.5)
        self.assertEqual(m["retrieval_recall"], 0.5)
        self.assertEqual(m["retrieval_f1"], 0.5)
        self.assertEqual(m["mrr"], 1.0)
        self.assertEqual(m["num_retrieved_chunks"], 2)

    def test_unanswerable_question_retrieval_metrics(self):
        retrieved = [12, 13, 14]
        relevant = []  # No relevant chunks in document
        m = calculate_retrieval_metrics(retrieved, relevant, k=3)

        self.assertEqual(m["retrieval_precision"], 0.0)
        self.assertIsNone(m["retrieval_recall"])
        self.assertIsNone(m["retrieval_f1"])
        self.assertIsNone(m["mrr"])
        self.assertIsNone(m["ndcg"])

    def test_abstention_evaluation(self):
        # Valid abstention phrases
        self.assertTrue(evaluate_abstention("The document does not provide enough information to answer.")[0])
        self.assertTrue(evaluate_abstention("This detail is not mentioned in the uploaded policy.")[0])
        self.assertTrue(evaluate_abstention("The exact value cannot be determined from the given context.")[0])

        # Non-abstaining assertion
        self.assertFalse(evaluate_abstention("The system achieved a 95% accuracy improvement.")[0])

    def test_citation_evaluation(self):
        # Answer with Page 1 citation
        res = evaluate_citations("The model is Titan V2 as shown on Page 1.", [12], [12])
        self.assertEqual(res["citation_precision"], 1.0)
        self.assertEqual(res["citation_coverage"], 1.0)
        self.assertTrue(res["has_page_citation"])

        # Answer with hallucinated Page 4 citation (resume only has Page 1)
        res_halluc = evaluate_citations("According to Page 4, the model is Titan V2.", [12], [12])
        self.assertEqual(res_halluc["citation_precision"], 0.5)


class TestAnswerQualityEvaluation(unittest.TestCase):
    """Tests evaluation of answer correctness, fact precision, recall, and hallucination."""

    def test_perfect_factual_answer(self):
        q_info = {
            "question_id": "Q01",
            "is_answerable": True,
            "expected_facts": ["Amazon Titan Text Embeddings V2 is used for embeddings"],
            "relevant_chunk_ids": [12, 13],
        }
        chunks = [
            {"chunk_id": 12, "page_number": 1, "text": "Amazon Titan Text Embeddings V2 for document and query embeddings"},
            {"chunk_id": 13, "page_number": 1, "text": "Amazon Bedrock Nova 2 Lite"},
        ]
        ans = "The Insurance AI Assistant uses Amazon Titan Text Embeddings V2 for embeddings, as detailed on Page 1."

        res = evaluate_question_answer(q_info, ans, chunks, temperature=0.0, top_k=2)
        self.assertEqual(res["answer_recall"], 1.0)
        self.assertEqual(res["answer_precision"], 1.0)
        self.assertEqual(res["answer_f1"], 1.0)
        self.assertEqual(res["faithfulness_score"], 1.0)
        self.assertEqual(res["hallucination_rate"], 0.0)
        self.assertTrue(res["abstention_correct"])

    def test_unanswerable_question_proper_abstention(self):
        q_info = {
            "question_id": "Q13",
            "is_answerable": False,
            "expected_facts": ["The document does not provide information on percentage improvement"],
            "relevant_chunk_ids": [],
        }
        chunks = [
            {"chunk_id": 12, "page_number": 1, "text": "FastAPI application in Python"},
        ]
        ans = "The document does not provide enough information to determine the percentage improvement in accuracy."

        res = evaluate_question_answer(q_info, ans, chunks, temperature=0.0, top_k=1)
        self.assertTrue(res["abstention_correct"])
        self.assertEqual(res["hallucination_rate"], 0.0)
        self.assertEqual(res["answer_f1"], 1.0)

    def test_unanswerable_question_hallucination_detected(self):
        q_info = {
            "question_id": "Q13",
            "is_answerable": False,
            "expected_facts": ["The document does not provide information on percentage improvement"],
            "relevant_chunk_ids": [],
        }
        chunks = [
            {"chunk_id": 12, "page_number": 1, "text": "FastAPI application in Python"},
        ]
        ans = "The Insurance AI Assistant achieved a 94.5% improvement in accuracy over keyword search."

        res = evaluate_question_answer(q_info, ans, chunks, temperature=0.3, top_k=1)
        self.assertFalse(res["abstention_correct"])
        self.assertEqual(res["hallucination_rate"], 1.0)
        self.assertTrue(len(res["unsupported_claims"]) > 0)


if __name__ == "__main__":
    unittest.main()
