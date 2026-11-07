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


DATA_PATH = Path("evaluation_results/ALL_ANSWERS.json")
OUTPUT_PATH = Path("evaluation_results/ragas_batch_results.json")

TARGETS = {
    ("Q01", "synthetic_insurance_policy_1_page.pdf"),
    ("Q01", "synthetic_insurance_policy_7_pages.pdf"),
    ("Q13", "synthetic_insurance_policy_1_page.pdf"),
    ("Q13", "synthetic_insurance_policy_6_pages.pdf"),
    ("Q11", "synthetic_insurance_policy_1_page.pdf"),
    ("Q11", "synthetic_insurance_policy_7_pages.pdf"),
}


def load_samples():
    with DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    selected = {}

    for record in data["matrix"]:
        key = (record.get("question_id"), record.get("document"))

        if key not in TARGETS:
            continue

        if record.get("status") != "success":
            continue

        if float(record.get("temperature", -1)) != 0.0:
            continue

        if int(record.get("retrieval_top_k", -1)) != 3:
            continue

        chunks = record.get("retrieved_chunks") or []
        contexts = [
            chunk["text"]
            for chunk in chunks
            if isinstance(chunk, dict) and chunk.get("text")
        ]

        if not contexts:
            continue

        selected[key] = {
            "user_input": record["question"],
            "response": record["generated_answer"],
            "reference": record["reference_answer"],
            "retrieved_contexts": contexts,
            "question_id": key[0],
            "document": key[1],
        }

    missing = TARGETS - set(selected)

    if missing:
        raise ValueError(f"Could not find these evaluation records: {missing}")

    return list(selected.values())


async def main():
    print("Loading six representative evaluation samples...")
    samples = load_samples()
    print(f"Loaded {len(samples)} samples.")

    print("Initializing Ragas models...")
    _, evaluator_llm, evaluator_embeddings = build_ragas_models()

    metrics = {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=3,
        ),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }

    results = []

    for index, sample in enumerate(samples, start=1):
        print(
            f"\n[{index}/{len(samples)}] "
            f"{sample['question_id']} - {sample['document']}"
        )

        row = {
            "question_id": sample["question_id"],
            "document": sample["document"],
            "scores": {},
            "errors": {},
        }

        for metric_name, metric in metrics.items():
            try:
                if metric_name == "faithfulness":
                    result = await metric.ascore(
                        user_input=sample["user_input"],
                        # pyrefly: ignore[unexpected-keyword]
                        response=sample["response"],
                        # pyrefly: ignore[unexpected-keyword]
                        retrieved_contexts=sample["retrieved_contexts"],
                    )

                elif metric_name == "answer_relevancy":
                    result = await metric.ascore(
                        user_input=sample["user_input"],
                        # pyrefly: ignore[unexpected-keyword]
                        response=sample["response"],
                    )

                elif metric_name == "context_precision":
                    result = await metric.ascore(
                        user_input=sample["user_input"],
                        # pyrefly: ignore[unexpected-keyword]
                        reference=sample["reference"],
                        # pyrefly: ignore[unexpected-keyword]
                        retrieved_contexts=sample["retrieved_contexts"],
                    )

                else:
                    result = await metric.ascore(
                        user_input=sample["user_input"],
                        # pyrefly: ignore[unexpected-keyword]
                        retrieved_contexts=sample["retrieved_contexts"],
                        # pyrefly: ignore[unexpected-keyword]
                        reference=sample["reference"],
                    )

                score = float(getattr(result, "value", result))
                row["scores"][metric_name] = score
                print(f"  {metric_name}: {score:.4f}")

            except Exception as exc:
                row["errors"][metric_name] = str(exc)
                print(f"  {metric_name}: FAILED - {exc}")

        results.append(row)

    output = {
        "sample_count": len(results),
        "evaluation_metrics": list(metrics.keys()),
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
