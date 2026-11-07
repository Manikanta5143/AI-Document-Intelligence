
import asyncio
import json
from pathlib import Path

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from scripts.ragas_evaluator import build_ragas_models


ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT_DIR / "evaluation_results" / "ALL_ANSWERS.json"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "evaluation_results"
    / "ragas_single_sample_result.json"
)


def load_sample():
    """Load one successful experiment with real RAG outputs."""

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    records = data.get("matrix", [])

    for record in records:

        if record.get("status") != "success":
            continue

        if not record.get("generated_answer"):
            continue

        if not record.get("reference_answer"):
            continue

        chunks = record.get("retrieved_chunks", [])

        contexts = [
            chunk["text"]
            for chunk in chunks
            if isinstance(chunk, dict)
            and isinstance(chunk.get("text"), str)
            and chunk["text"].strip()
        ]

        if not contexts:
            continue

        sample = {
            "user_input": record["question"],
            "response": record["generated_answer"],
            "retrieved_contexts": contexts,
            "reference": record["reference_answer"],
            "question_id": record.get("question_id"),
            "document": record.get("document"),
            "temperature": record.get("temperature"),
            "top_k": record.get("retrieval_top_k"),
        }

        return sample

    raise ValueError("No suitable successful evaluation record found.")


async def main():

    print("Loading existing RAG evaluation data...")

    sample = load_sample()

    print("\nSelected experiment:")
    print("Document:", sample["document"])
    print("Question ID:", sample["question_id"])
    print("Question:", sample["user_input"])
    print("Retrieved chunks:", len(sample["retrieved_contexts"]))

    print("\nInitializing Ragas evaluator models...")

    settings, evaluator_llm, evaluator_embeddings = (
        build_ragas_models()
    )

    print("Evaluator model:", settings.bedrock_model_id)
    print("Region:", settings.aws_region)

    # Initialize the four Ragas metrics

    metrics = {
        "faithfulness": Faithfulness(
            llm=evaluator_llm
        ),

        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=3,
        ),

        "context_precision": ContextPrecision(
            llm=evaluator_llm
        ),

        "context_recall": ContextRecall(
            llm=evaluator_llm
        ),
    }

    print("\nStarting Ragas evaluation...")

    # 1. Faithfulness
    faithfulness_result = await metrics[
        "faithfulness"
    ].ascore(
        user_input=sample["user_input"],
        response=sample["response"],
        retrieved_contexts=sample["retrieved_contexts"],
    )

    # 2. Answer Relevancy
    relevancy_result = await metrics[
        "answer_relevancy"
    ].ascore(
        user_input=sample["user_input"],
        response=sample["response"],
    )

    # 3. Context Precision
    precision_result = await metrics[
        "context_precision"
    ].ascore(
        user_input=sample["user_input"],
        reference=sample["reference"],
        retrieved_contexts=sample["retrieved_contexts"],
    )

    # 4. Context Recall
    recall_result = await metrics[
        "context_recall"
    ].ascore(
        user_input=sample["user_input"],
        retrieved_contexts=sample["retrieved_contexts"],
        reference=sample["reference"],
    )

    results = {
        "metadata": {
            "question_id": sample["question_id"],
            "document": sample["document"],
            "temperature": sample["temperature"],
            "top_k": sample["top_k"],
            "evaluator_model": settings.bedrock_model_id,
            "embedding_model": settings.bedrock_embedding_model_id,
        },

        "scores": {
            "faithfulness": faithfulness_result.value,
            "answer_relevancy": relevancy_result.value,
            "context_precision": precision_result.value,
            "context_recall": recall_result.value,
        }
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== RAGAS RESULTS ==========")

    for name, score in results["scores"].items():
        print(f"{name}: {score:.4f}")

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(main())
