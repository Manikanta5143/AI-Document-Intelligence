"""Comprehensive RAG Evaluation and Hyperparameter Optimization Runner.

Executes:
- Retrieval evaluation across top_k in [1, 2, 3, 4, 5]
- Live Amazon Bedrock Nova 2 Lite generation across temperatures in [0.0, 0.1, 0.2, 0.3]
- 20 unique parameter configurations x 15 questions = 300 primary evaluations (run_1)
- Repeated generation runs for temperatures > 0.0 at top_k=3 = 45 repeat evaluations (run_2)
- Total 345 live model evaluations
- Aggregates metrics, generates 8 visualization charts, and outputs all required reports.
"""
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from app.aws.llm_service import BedrockLLMService
from app.config import get_settings
from app.database.session import get_db
from app.rag.vector_store import get_vector_store_service
from app.repositories.document_repository import DocumentRepository
from scripts.rag_eval_engine import (
    calculate_retrieval_metrics,
    evaluate_question_answer,
)

EVAL_DIR = ROOT_DIR / "evaluation_results"
PLOTS_DIR = EVAL_DIR / "plots"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Nova Lite Pricing in ap-south-1 (per 1,000 tokens)
NOVA_LITE_INPUT_COST_PER_1K = 0.00006   # $0.06 per 1M tokens
NOVA_LITE_OUTPUT_COST_PER_1K = 0.00024  # $0.24 per 1M tokens


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1000.0 * NOVA_LITE_INPUT_COST_PER_1K) + (
        output_tokens / 1000.0 * NOVA_LITE_OUTPUT_COST_PER_1K
    )


def invoke_bedrock_with_retry(
    llm: BedrockLLMService,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """Invoke Bedrock Nova Lite with exponential backoff retry."""
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            res = llm.generate_text_with_metadata(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            wall_latency_ms = (time.time() - t0) * 1000.0
            # If server didn't report latency, use wall time
            if res.get("latency_ms", 0) <= 0:
                res["latency_ms"] = wall_latency_ms
            return res
        except Exception as e:
            print(f"  [Attempt {attempt}/{max_retries}] Bedrock error: {e}. Retrying in {delay}s...")
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2.0
    raise RuntimeError("Bedrock invocation failed after max retries")


def main():
    print("=" * 80)
    print("STARTING COMPREHENSIVE RAG EVALUATION & HYPERPARAMETER OPTIMIZATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    # 1. Load Ground Truth
    gt_path = EVAL_DIR / "question_ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    questions = gt_data["questions"]
    print(f"Loaded {len(questions)} verified evaluation questions.")

    # 2. Setup Services & Verify Target Document
    db = next(get_db())
    doc_repo = DocumentRepository(db)
    target_doc = doc_repo.get_by_id(3)
    if not target_doc:
        raise RuntimeError("Evaluation document ID 3 ('Manikanta_Ausula_Resume.pdf') not found in database!")

    print(f"Verified evaluation document: ID={target_doc.id}, Name='{target_doc.file_name}', Status='{target_doc.status}'")

    vs = get_vector_store_service()
    llm = BedrockLLMService()
    print(f"Vector Store Collection Count: {vs.get_collection_count()}")
    print(f"Bedrock LLM Model: {llm.model_id} (Region: {llm.settings.aws_region})")

    # System instruction (exact prompt from production RAGService)
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

    # 3. Phase 1: Pre-compute and Cache Retrieval for all (question, top_k)
    print("\n" + "-" * 60)
    print("[PHASE 1] Pre-computing and caching retrieval for 15 questions x 5 top_k values...")
    print("-" * 60)

    retrieval_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
    top_k_values = [1, 2, 3, 4, 5]
    temperatures = [0.0, 0.1, 0.2, 0.3]

    retrieval_records: List[Dict[str, Any]] = []

    for q in questions:
        qid = q["question_id"]
        q_text = q["question"]
        gt_relevant_ids = q["relevant_chunk_ids"]

        for k in top_k_values:
            t_start = time.time()
            chunks = vs.similarity_search(query=q_text, top_k=k, document_id=target_doc.id)
            retrieval_latency = (time.time() - t_start) * 1000.0

            retrieved_chunk_ids = [c["chunk_id"] for c in chunks]

            ret_metrics = calculate_retrieval_metrics(
                retrieved_chunk_ids=retrieved_chunk_ids,
                ground_truth_relevant_ids=gt_relevant_ids,
                k=k,
                latency_ms=retrieval_latency,
            )

            # Build excerpt context
            context_blocks = [
                f"[Excerpt {i+1} | Page {c['page_number']} | Chunk ID: {c['chunk_id']}]\n{c['text']}"
                for i, c in enumerate(chunks)
            ]
            retrieved_context_text = "\n\n".join(context_blocks)

            user_prompt = (
                f"Document Name: {target_doc.file_name}\n"
                f"Document Category: {target_doc.document_type}\n\n"
                f"Retrieved Document Context:\n"
                f"{retrieved_context_text}\n\n"
                f"User Question: {q_text}\n\n"
                f"Answer:"
            )

            retrieval_cache[(qid, k)] = {
                "chunks": chunks,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_context_text": retrieved_context_text,
                "user_prompt": user_prompt,
                "metrics": ret_metrics,
            }

            retrieval_records.append({
                "question_id": qid,
                "category": q["category"],
                "top_k": k,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "ground_truth_relevant_ids": gt_relevant_ids,
                **ret_metrics,
            })

    print(f"Successfully cached {len(retrieval_cache)} retrieval pairs.")

    # 4. Phase 2 & 3: Run Full Experiment Matrix
    print("\n" + "-" * 60)
    print("[PHASE 2 & 3] Executing Live Amazon Bedrock Generations across 20 configurations...")
    print("-" * 60)

    all_results: List[Dict[str, Any]] = []

    # Intermediate checkpoint file
    checkpoint_csv = EVAL_DIR / "all_experiment_results.csv"
    checkpoint_json = EVAL_DIR / "all_experiment_results.json"

    eval_counter = 0
    total_evals_planned = (len(temperatures) * len(top_k_values) * len(questions)) + (3 * len(questions))
    print(f"Planned total evaluations: 300 primary (run_1) + 45 repeat (run_2) = {total_evals_planned}")

    # Primary Run: run_1 for all 20 configurations
    for temp in temperatures:
        for k in top_k_values:
            cfg_label = f"temp_{temp}_topk_{k}"
            print(f"\n>>> Running Configuration: Temperature={temp}, top_k={k} (run_1) <<<")

            for q in questions:
                eval_counter += 1
                qid = q["question_id"]
                cache_entry = retrieval_cache[(qid, k)]
                user_prompt = cache_entry["user_prompt"]
                ret_metrics = cache_entry["metrics"]
                retrieved_chunks = cache_entry["chunks"]

                print(f"[{eval_counter}/{total_evals_planned}] {cfg_label} | {qid}...", end="", flush=True)

                bedrock_res = invoke_bedrock_with_retry(
                    llm=llm,
                    prompt=user_prompt,
                    system_prompt=system_instruction,
                    temperature=temp,
                )

                gen_answer = bedrock_res["text"]
                in_tokens = bedrock_res["input_tokens"]
                out_tokens = bedrock_res["output_tokens"]
                latency_ms = bedrock_res["latency_ms"]
                cost = compute_cost(in_tokens, out_tokens)

                ans_metrics = evaluate_question_answer(
                    question_info=q,
                    generated_answer=gen_answer,
                    retrieved_chunks=retrieved_chunks,
                    temperature=temp,
                    top_k=k,
                )

                record = {
                    "question_id": qid,
                    "question": q["question"],
                    "question_category": q["category"],
                    "category_code": q["category_code"],
                    "temperature": temp,
                    "retrieval_top_k": k,
                    "run_id": "run_1",
                    "reference_answer": q["reference_answer"],
                    "retrieved_chunk_ids": cache_entry["retrieved_chunk_ids"],
                    "retrieved_source_text": cache_entry["retrieved_context_text"],
                    "generated_answer": gen_answer,
                    "expected_facts": ans_metrics["expected_facts"],
                    "supported_facts": ans_metrics["supported_facts"],
                    "missing_facts": ans_metrics["missing_facts"],
                    "unsupported_claims": ans_metrics["unsupported_claims"],
                    "contradictions": ans_metrics["contradictions"],
                    "retrieval_precision": ret_metrics["retrieval_precision"],
                    "retrieval_recall": ret_metrics["retrieval_recall"],
                    "retrieval_f1": ret_metrics["retrieval_f1"],
                    "mrr": ret_metrics["mrr"],
                    "ndcg": ret_metrics["ndcg"],
                    "context_relevance": ret_metrics["context_relevance"],
                    "context_coverage": ret_metrics["context_coverage"],
                    "answer_precision": ans_metrics["answer_precision"],
                    "answer_recall": ans_metrics["answer_recall"],
                    "answer_f1": ans_metrics["answer_f1"],
                    "faithfulness_score": ans_metrics["faithfulness_score"],
                    "groundedness_score": ans_metrics["groundedness_score"],
                    "answer_correctness_score": ans_metrics["answer_correctness_score"],
                    "completeness_score": ans_metrics["completeness_score"],
                    "hallucination_rate": ans_metrics["hallucination_rate"],
                    "citation_precision": ans_metrics["citation_precision"],
                    "citation_coverage": ans_metrics["citation_coverage"],
                    "abstention_correct": ans_metrics["abstention_correct"],
                    "latency_ms": round(latency_ms, 2),
                    "retrieval_latency_ms": ret_metrics["retrieval_latency_ms"],
                    "total_latency_ms": round(latency_ms + ret_metrics["retrieval_latency_ms"], 2),
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "total_tokens": in_tokens + out_tokens,
                    "estimated_cost": round(cost, 6),
                    "evaluation_notes": ans_metrics["evaluation_notes"],
                }
                all_results.append(record)

                print(f" Done ({latency_ms:.0f}ms, F1={ans_metrics['answer_f1']:.2f}, in={in_tokens}, out={out_tokens})")
                time.sleep(0.1)

    # Repeat Run: run_2 for temperatures > 0.0 at top_k=3 to measure consistency
    print("\n" + "-" * 60)
    print("[REPEATABILITY RUN] Executing run_2 for temp in [0.1, 0.2, 0.3] at top_k=3...")
    print("-" * 60)
    for temp in [0.1, 0.2, 0.3]:
        k = 3
        cfg_label = f"temp_{temp}_topk_{k}_run2"
        for q in questions:
            eval_counter += 1
            qid = q["question_id"]
            cache_entry = retrieval_cache[(qid, k)]
            user_prompt = cache_entry["user_prompt"]
            ret_metrics = cache_entry["metrics"]
            retrieved_chunks = cache_entry["chunks"]

            print(f"[{eval_counter}/{total_evals_planned}] {cfg_label} | {qid}...", end="", flush=True)

            bedrock_res = invoke_bedrock_with_retry(
                llm=llm,
                prompt=user_prompt,
                system_prompt=system_instruction,
                temperature=temp,
            )

            gen_answer = bedrock_res["text"]
            in_tokens = bedrock_res["input_tokens"]
            out_tokens = bedrock_res["output_tokens"]
            latency_ms = bedrock_res["latency_ms"]
            cost = compute_cost(in_tokens, out_tokens)

            ans_metrics = evaluate_question_answer(
                question_info=q,
                generated_answer=gen_answer,
                retrieved_chunks=retrieved_chunks,
                temperature=temp,
                top_k=k,
            )

            record = {
                "question_id": qid,
                "question": q["question"],
                "question_category": q["category"],
                "category_code": q["category_code"],
                "temperature": temp,
                "retrieval_top_k": k,
                "run_id": "run_2",
                "reference_answer": q["reference_answer"],
                "retrieved_chunk_ids": cache_entry["retrieved_chunk_ids"],
                "retrieved_source_text": cache_entry["retrieved_context_text"],
                "generated_answer": gen_answer,
                "expected_facts": ans_metrics["expected_facts"],
                "supported_facts": ans_metrics["supported_facts"],
                "missing_facts": ans_metrics["missing_facts"],
                "unsupported_claims": ans_metrics["unsupported_claims"],
                "contradictions": ans_metrics["contradictions"],
                "retrieval_precision": ret_metrics["retrieval_precision"],
                "retrieval_recall": ret_metrics["retrieval_recall"],
                "retrieval_f1": ret_metrics["retrieval_f1"],
                "mrr": ret_metrics["mrr"],
                "ndcg": ret_metrics["ndcg"],
                "context_relevance": ret_metrics["context_relevance"],
                "context_coverage": ret_metrics["context_coverage"],
                "answer_precision": ans_metrics["answer_precision"],
                "answer_recall": ans_metrics["answer_recall"],
                "answer_f1": ans_metrics["answer_f1"],
                "faithfulness_score": ans_metrics["faithfulness_score"],
                "groundedness_score": ans_metrics["groundedness_score"],
                "answer_correctness_score": ans_metrics["answer_correctness_score"],
                "completeness_score": ans_metrics["completeness_score"],
                "hallucination_rate": ans_metrics["hallucination_rate"],
                "citation_precision": ans_metrics["citation_precision"],
                "citation_coverage": ans_metrics["citation_coverage"],
                "abstention_correct": ans_metrics["abstention_correct"],
                "latency_ms": round(latency_ms, 2),
                "retrieval_latency_ms": ret_metrics["retrieval_latency_ms"],
                "total_latency_ms": round(latency_ms + ret_metrics["retrieval_latency_ms"], 2),
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens,
                "estimated_cost": round(cost, 6),
                "evaluation_notes": ans_metrics["evaluation_notes"],
            }
            all_results.append(record)

            print(f" Done ({latency_ms:.0f}ms, F1={ans_metrics['answer_f1']:.2f}, in={in_tokens}, out={out_tokens})")
            time.sleep(0.1)

    print(f"\nAll {len(all_results)} evaluations completed successfully!")

    # 5. Save all_experiment_results.json and all_experiment_results.csv
    with open(checkpoint_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved full JSON dataset to {checkpoint_json}")

    # For CSV, serialize lists as json strings
    csv_rows = []
    for r in all_results:
        row_copy = dict(r)
        for list_key in [
            "retrieved_chunk_ids",
            "expected_facts",
            "supported_facts",
            "missing_facts",
            "unsupported_claims",
            "contradictions",
        ]:
            row_copy[list_key] = json.dumps(row_copy[list_key])
        csv_rows.append(row_copy)

    df_all = pd.DataFrame(csv_rows)
    df_all.to_csv(checkpoint_csv, index=False, encoding="utf-8")
    print(f"Saved full CSV dataset to {checkpoint_csv} ({len(df_all)} rows)")

    # 6. Aggregate Metrics Calculation
    print("\n" + "-" * 60)
    print("[PHASE 4] Calculating Aggregate Metrics per Configuration...")
    print("-" * 60)

    # Use run_1 for primary matrix comparison across 20 configurations
    df_run1 = pd.DataFrame(all_results)
    df_run1 = df_run1[df_run1["run_id"] == "run_1"]

    # Filter answerable questions for retrieval metrics aggregation
    # Unanswerable questions have None retrieval metrics
    agg_rows = []
    for temp in temperatures:
        for k in top_k_values:
            subset = df_run1[(df_run1["temperature"] == temp) & (df_run1["retrieval_top_k"] == k)]
            ans_subset = subset[subset["category_code"] != "E"]

            ret_prec = ans_subset["retrieval_precision"].mean()
            ret_rec = ans_subset["retrieval_recall"].mean()
            ret_f1 = ans_subset["retrieval_f1"].mean()
            ret_mrr = ans_subset["mrr"].mean()
            ret_ndcg = ans_subset["ndcg"].mean()

            ans_prec = subset["answer_precision"].mean()
            ans_rec = subset["answer_recall"].mean()
            ans_f1 = subset["answer_f1"].mean()
            faith = subset["faithfulness_score"].mean()
            ground = subset["groundedness_score"].mean()
            corr = subset["answer_correctness_score"].mean()
            comp = subset["completeness_score"].mean()
            halluc = subset["hallucination_rate"].mean()
            cit_prec = subset["citation_precision"].mean()
            cit_cov = subset["citation_coverage"].mean()
            abstain_acc = subset["abstention_correct"].astype(float).mean()

            lat_avg = subset["latency_ms"].mean()
            lat_med = subset["latency_ms"].median()
            lat_std = subset["latency_ms"].std()

            cost_mean = subset["estimated_cost"].mean()
            cost_total = subset["estimated_cost"].sum()

            agg_rows.append({
                "temperature": temp,
                "retrieval_top_k": k,
                "configuration": f"T={temp}, top_k={k}",
                "mean_retrieval_precision": round(ret_prec, 4),
                "mean_retrieval_recall": round(ret_rec, 4),
                "mean_retrieval_f1": round(ret_f1, 4),
                "mean_mrr": round(ret_mrr, 4),
                "mean_ndcg": round(ret_ndcg, 4),
                "mean_answer_precision": round(ans_prec, 4),
                "mean_answer_recall": round(ans_rec, 4),
                "mean_answer_f1": round(ans_f1, 4),
                "mean_faithfulness": round(faith, 4),
                "mean_groundedness": round(ground, 4),
                "mean_correctness": round(corr, 4),
                "mean_completeness": round(comp, 4),
                "hallucination_rate": round(halluc, 4),
                "citation_precision": round(cit_prec, 4),
                "citation_coverage": round(cit_cov, 4),
                "abstention_accuracy": round(abstain_acc, 4),
                "mean_latency_ms": round(lat_avg, 2),
                "median_latency_ms": round(lat_med, 2),
                "std_latency_ms": round(lat_std, 2),
                "mean_cost_usd": round(cost_mean, 6),
                "total_cost_usd": round(cost_total, 6),
            })

    df_agg = pd.DataFrame(agg_rows)
    agg_csv = EVAL_DIR / "aggregate_metrics.csv"
    df_agg.to_csv(agg_csv, index=False, encoding="utf-8")
    print(f"Saved aggregate metrics to {agg_csv} ({len(df_agg)} configurations)")

    # 7. Generate Visualizations (Phase 5)
    print("\n" + "-" * 60)
    print("[PHASE 5] Generating Visualizations in evaluation_results/plots/...")
    print("-" * 60)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"

    # Chart 1: Retrieval Precision vs top_k
    plt.figure(figsize=(8, 5))
    ret_summary = df_agg.groupby("retrieval_top_k")["mean_retrieval_precision"].mean().reset_index()
    plt.plot(ret_summary["retrieval_top_k"], ret_summary["mean_retrieval_precision"], marker="o", linewidth=2.5, color="#1f77b4")
    plt.title("Retrieval Precision@k vs. Retrieval top_k", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Retrieval top_k", fontsize=12)
    plt.ylabel("Mean Retrieval Precision@k", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(top_k_values)
    for _, r in ret_summary.iterrows():
        plt.annotate(f"{r['mean_retrieval_precision']:.3f}", (r["retrieval_top_k"], r["mean_retrieval_precision"] + 0.03), ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "retrieval_precision_vs_topk.png", dpi=300)
    plt.close()
    print("  -> Saved plots/retrieval_precision_vs_topk.png")

    # Chart 2: Retrieval Recall vs top_k
    plt.figure(figsize=(8, 5))
    rec_summary = df_agg.groupby("retrieval_top_k")["mean_retrieval_recall"].mean().reset_index()
    plt.plot(rec_summary["retrieval_top_k"], rec_summary["mean_retrieval_recall"], marker="s", linewidth=2.5, color="#2ca02c")
    plt.title("Retrieval Recall@k vs. Retrieval top_k", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Retrieval top_k", fontsize=12)
    plt.ylabel("Mean Retrieval Recall@k", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(top_k_values)
    for _, r in rec_summary.iterrows():
        plt.annotate(f"{r['mean_retrieval_recall']:.3f}", (r["retrieval_top_k"], r["mean_retrieval_recall"] + 0.03), ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "retrieval_recall_vs_topk.png", dpi=300)
    plt.close()
    print("  -> Saved plots/retrieval_recall_vs_topk.png")

    # Chart 3: Retrieval F1 vs top_k
    plt.figure(figsize=(8, 5))
    f1_summary = df_agg.groupby("retrieval_top_k")["mean_retrieval_f1"].mean().reset_index()
    plt.plot(f1_summary["retrieval_top_k"], f1_summary["mean_retrieval_f1"], marker="^", linewidth=2.5, color="#d62728")
    plt.title("Retrieval F1@k vs. Retrieval top_k", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Retrieval top_k", fontsize=12)
    plt.ylabel("Mean Retrieval F1@k", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(top_k_values)
    for _, r in f1_summary.iterrows():
        plt.annotate(f"{r['mean_retrieval_f1']:.3f}", (r["retrieval_top_k"], r["mean_retrieval_f1"] + 0.03), ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "retrieval_f1_vs_topk.png", dpi=300)
    plt.close()
    print("  -> Saved plots/retrieval_f1_vs_topk.png")

    # Chart 4: Answer Correctness vs Temperature
    plt.figure(figsize=(8, 5))
    temp_corr = df_agg.groupby("temperature")["mean_correctness"].mean().reset_index()
    sns.barplot(data=temp_corr, x="temperature", y="mean_correctness", palette="Blues_d")
    plt.title("Answer Correctness vs. Generation Temperature", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Mean Answer Correctness Score", fontsize=12)
    plt.ylim(0, 1.05)
    for idx, r in temp_corr.iterrows():
        plt.annotate(f"{r['mean_correctness']:.3f}", (idx, r["mean_correctness"] + 0.03), ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "answer_correctness_vs_temperature.png", dpi=300)
    plt.close()
    print("  -> Saved plots/answer_correctness_vs_temperature.png")

    # Chart 5: Answer F1 vs Temperature
    plt.figure(figsize=(8, 5))
    temp_f1 = df_agg.groupby("temperature")["mean_answer_f1"].mean().reset_index()
    sns.lineplot(data=df_agg, x="temperature", y="mean_answer_f1", hue="retrieval_top_k", marker="o", palette="tab10")
    plt.title("Answer F1 vs. Temperature across top_k Configurations", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Mean Answer F1 Score", fontsize=12)
    plt.ylim(0.4, 1.02)
    plt.legend(title="Retrieval top_k", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "answer_f1_vs_temperature.png", dpi=300)
    plt.close()
    print("  -> Saved plots/answer_f1_vs_temperature.png")

    # Chart 6: Faithfulness and Hallucination Rate for all configurations
    plt.figure(figsize=(12, 6))
    x_indices = np.arange(len(df_agg))
    width = 0.35
    plt.bar(x_indices - width / 2, df_agg["mean_faithfulness"], width, label="Faithfulness", color="#2ca02c")
    plt.bar(x_indices + width / 2, df_agg["hallucination_rate"], width, label="Hallucination Rate", color="#d62728")
    plt.title("Faithfulness vs. Hallucination Rate across All 20 Configurations", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Configuration (T=Temp, k=top_k)", fontsize=11)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.xticks(x_indices, [f"T{r['temperature']}/k{r['retrieval_top_k']}" for _, r in df_agg.iterrows()], rotation=45, ha="right", fontsize=9)
    plt.ylim(0, 1.15)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "faithfulness_and_hallucination_rate.png", dpi=300)
    plt.close()
    print("  -> Saved plots/faithfulness_and_hallucination_rate.png")

    # Chart 7: Latency and Cost vs Configuration
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.plot(df_agg["configuration"], df_agg["mean_latency_ms"], color="#1f77b4", marker="o", linewidth=2, label="Latency (ms)")
    ax2.plot(df_agg["configuration"], df_agg["mean_cost_usd"] * 1000, color="#ff7f0e", marker="s", linewidth=2, linestyle="--", label="Cost (x10^-3 USD)")
    ax1.set_xlabel("Configuration", fontsize=12)
    ax1.set_ylabel("Response Latency (ms)", color="#1f77b4", fontsize=12)
    ax2.set_ylabel("Cost per Query ($ x 10^-3 USD)", color="#ff7f0e", fontsize=12)
    ax1.set_xticklabels(df_agg["configuration"], rotation=45, ha="right", fontsize=9)
    plt.title("Latency and Cost per Query across Configurations", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    plt.savefig(PLOTS_DIR / "latency_and_cost_vs_configuration.png", dpi=300)
    plt.close()
    print("  -> Saved plots/latency_and_cost_vs_configuration.png")

    # Chart 8: Heatmap of Answer F1 for Temperature and top_k combinations
    plt.figure(figsize=(8, 6))
    pivot_f1 = df_agg.pivot(index="temperature", columns="retrieval_top_k", values="mean_answer_f1")
    sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Answer F1 Score"})
    plt.title("Heatmap of Answer F1 Score by Temperature and Retrieval top_k", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Retrieval top_k", fontsize=12)
    plt.ylabel("Temperature", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "heatmap_answer_f1.png", dpi=300)
    plt.close()
    print("  -> Saved plots/heatmap_answer_f1.png")

    print("\n" + "=" * 80)
    print("EXPERIMENT EXECUTION COMPLETE!")
    print(f"Total Model Calls: {len(all_results)}")
    print(f"Total Test Cost: ${sum(r['estimated_cost'] for r in all_results):.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
