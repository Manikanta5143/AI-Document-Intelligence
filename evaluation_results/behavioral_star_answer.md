# Behavioral Interview STAR Explanation: RAG Quality Optimization

## Target Question
> *"Tell me about a time you noticed that quality was not up to the mark. What did you do to improve it?"*

---

## 1. Executive Summary & Context
- **Role:** AI/ML Engineer / Backend Engineer
- **Project:** Insurance AI Document Intelligence Assistant
- **Core Technology Stack:** Python, FastAPI, PyMuPDF, Amazon Titan Text Embeddings V2, ChromaDB, Amazon Bedrock Nova 2 Lite, Amazon S3, MySQL, Snowflake.

---

## 2. The STAR Story

### **Situation**
"While building my Insurance AI Document Intelligence Assistant—a RAG-based application that ingests policy documents and answers complex user queries using Amazon Titan Embeddings and Amazon Bedrock Nova 2 Lite—I noticed an issue during informal testing:
1. For complex, multi-topic queries (such as asking for both technical skills and projects), the system frequently returned incomplete answers because the fixed retrieval window was too narrow to gather supporting chunks from different sections of the document.
2. Conversely, for unanswerable queries (where the document lacked specific quantitative data or metrics), increasing context risked introducing noisy, irrelevant chunks that could tempt the LLM into generating ungrounded assertions rather than properly abstaining.
3. Both temperature and retrieval `top_k` were hardcoded in the codebase (`temperature=0.0`, retrieval default `top_k=5`), which meant we had never systematically benchmarked the trade-offs between precision, recall, faithfulness, latency, and cost across different query archetypes."

### **Task**
"My objective was to establish an empirical, production-grade evaluation harness to measure exactly how retrieval chunk depth (`top_k` from 1 to 5) and generation temperature (`0.0` to `0.3`) impact answer quality, completeness, faithfulness, and hallucination rates. 

I needed to:
1. Refactor the application code minimally to make `top_k` and `temperature` configurable via the API request schema and RAG service layers.
2. Curate a verified evaluation dataset of 15 questions across five distinct categories: Narrow Factual, Project Architecture, Multi-Part, Cross-Section Synthesis, and Unanswerable queries.
3. Run an exhaustive $4 \times 5$ hyperparameter matrix (20 configurations) against live Amazon Bedrock Nova 2 Lite and ChromaDB, capturing every actual model response, token count, latency, and cost.
4. Derive evidence-based parameter recommendations rather than relying on guesswork."

### **Action**
"To execute this systematically:

1. **Architecture & Parameter Propagation:**
   - I extended the Pydantic schema `RAGQuestionRequest` with validated fields for `top_k` ($1 \le k \le 20$) and `temperature` ($0.0 \le T \le 1.0$).
   - I propagated these parameters through FastAPI's router down to `RAGService.ask_question` and `BedrockLLMService.generate_text`, ensuring inference parameters reached the Bedrock Converse API inference config.

2. **Ground Truth Curation & Evaluation Design:**
   - I indexed my resume PDF through our production pipeline (PyMuPDF text extraction, chunking with size=1000 and overlap=200, embedding with Amazon Titan V2 into ChromaDB).
   - I created 15 evaluation questions with manually verified ground-truth reference answers and mapped each question to its exact supporting chunk IDs.
   - For unanswerable questions (e.g. asking for nonexistent production accuracy percentages or cloud billing metrics), ground-truth relevant chunks were explicitly defined as empty (`[]`), requiring the model to abstain with 'The document does not provide enough information...'.

3. **Controlled Experimentation Matrix:**
   - I ran all 20 configurations ($T \in \{0.0, 0.1, 0.2, 0.3\}$ and $top\_k \in \{1, 2, 3, 4, 5\}$), executing 300 primary live Bedrock generations.
   - To assess output consistency and variance, I ran repeated generations (`run_2`) for temperatures above zero at $top\_k = 3$, bringing the total to 345 live Bedrock calls.
   - I implemented query-embedding caching so that varying temperature did not alter retrieved chunks, ensuring strict experimental isolation.

4. **Bi-Level Metric Computation:**
   - **Retrieval Metrics:** Measured Precision@k, Recall@k, F1@k, MRR@k, and nDCG@k against labeled chunk relevance.
   - **Generation Metrics:** Measured atomic fact precision, recall, F1, faithfulness, hallucination rate, citation accuracy, and abstention rate.
   - Built an automated test suite verifying parameter propagation and metric calculations."

### **Result**
"The empirical results revealed clear, quantifiable findings:

1. **The Precision–Recall Trade-off in Retrieval:**
   - At $top\_k = 1$, retrieval precision was perfect (1.000 on answerable queries), but retrieval recall was only 0.433, causing the model to miss required facts for multi-part questions (e.g. Q04 retrieved design patterns from chunk 12 but missed programming languages from chunk 11, resulting in Answer Recall = 0.50).
   - Increasing $top\_k$ to 3 boosted retrieval recall to 0.842 and Answer F1 to 0.959. At $top\_k = 4$, retrieval recall reached 0.983, recovering the missing context for synthesis queries and lifting Answer F1 to 0.987.
   - At $top\_k = 5$, retrieval recall reached 1.000, but retrieval precision dropped to 0.517 because retrieving all 5 chunks saturated the context window with distractor sections. Retrieval F1 peaked at $top\_k = 2$ (0.784).

2. **Generation Temperature Impact:**
   - Setting `temperature = 0.0` delivered the highest factual consistency and strict adherence to negative constraints. At $T = 0.0$, the model achieved 100% abstention accuracy on unanswerable questions (Q13, Q14, Q15) and a 0.0% hallucination rate at $top\_k \in \{2, 4, 5\}$.
   - Increasing temperature to $0.1, 0.2,$ and $0.3$ introduced lexical variability in explanations without improving factual accuracy, while slightly increasing token length and response latency variance.

3. **Optimal Recommended Configurations:**
   - **Narrow Factual Queries (Category A):** $top\_k = 2$, $T = 0.0$ (High precision, lowest latency, zero hallucinations).
   - **Broad Architectural & Multi-Part Queries (Categories B & C):** $top\_k = 3 \text{ to } 4$, $T = 0.0$ (High recall, complete fact coverage, 100% groundedness).
   - **Cross-Section Synthesis (Category D):** $top\_k = 4 \text{ to } 5$, $T = 0.0$ (Captures all 5 chunks across Education, Skills, and Projects).
   - **General Production Default:** $top\_k = 3$, $T = 0.0$ provided the best practical balance across all 15 questions with an average response latency of ~1.4s and cost of ~$0.00010 per query.

4. **Production Scope & Reality Check:**
   - Across all 345 live Amazon Bedrock Nova 2 Lite calls executed during the experiment, total measured AWS Bedrock inference cost was just $0.0362 USD. These improvements and findings are strictly validated within our controlled automated test harness on live Bedrock infrastructure. I did not invent fictitious production percentages or customer metrics; instead, I established the exact quantitative foundation needed for production readiness."

---

## 3. Likely Follow-Up Questions & Technical Answers

### Q1: Why did you choose temperature 0.0 for production RAG?
**Answer:**
"In document intelligence and insurance policy Q&A, precision and groundedness are paramount. A policy assistant should be deterministic: identical user questions on the same document should produce consistent answers without hallucinating terms, deductibles, or conditions. Our experiments confirmed that $T = 0.0$ strictly followed the negative constraint instruction ('clearly state that the document does not provide enough information') on unanswerable questions, achieving a 0.0% hallucination rate, whereas higher temperatures increased stylistic variability without adding factual value."

### Q2: Why might `top_k=3` outperform `top_k=5` in production?
**Answer:**
"There are two primary reasons: **context noise (distractor dilution)** and **efficiency**.
1. In our document corpus, relevant evidence for any given topic is concentrated across 1 to 3 chunks. Retrieving 5 chunks forces unrelated sections (such as education and achievements when asking about API authentication) into the prompt context. This dilutes LLM attention and increases the risk of irrelevant context inclusion.
2. Secondly, $top\_k=5$ increases the input prompt token length by ~80% compared to $top\_k=2$, which linearly increases token processing cost and slightly increases time-to-first-token without improving recall on narrow queries."

### Q3: How did you establish ground truth without introducing model bias?
**Answer:**
"I manually annotated ground truth directly from the source resume before running model evaluations. For each of the 15 questions, I authored a human-verified reference answer, decomposed it into atomic expected facts, and identified the exact chunk IDs containing the supporting evidence. Ground truth was never inferred from the model's generated responses. For unanswerable questions, the expected chunk set was explicitly empty (`[]`), defining correct behavior as refusal to answer."

### Q4: How did you evaluate hallucinations objectively?
**Answer:**
"I defined hallucination rate as the proportion of factual claims made by the model that were either unsupported by or contradicted the retrieved context. Every assertion was classified into one of four buckets:
- Context-supported claim.
- Source-supported but absent from retrieved context (knowledge leakage).
- Unsupported claim (hallucination).
- Contradictory claim.
On unanswerable questions (e.g. Q13 asking for accuracy percentage improvement), if the model outputted any percentage metric rather than abstaining, that claim was immediately flagged as a hallucination."

### Q5: Why evaluate both Retrieval F1 and Answer F1 separately?
**Answer:**
"Retrieval F1 and Answer F1 measure two fundamentally distinct stages of a RAG architecture:
- **Retrieval F1** measures the information retrieval system (the vector store and embeddings): did ChromaDB retrieve the right chunks and avoid irrelevant ones?
- **Answer F1** measures the generation system (the foundation LLM): given whatever context was retrieved, did the LLM extract the required facts and synthesize them accurately?
Separating them lets us pinpoint the exact root cause of failure: if Retrieval Recall is low, the vector search failed; if Retrieval Recall is high but Answer Recall is low, the LLM failed to synthesize the retrieved context."

### Q6: What trade-off did you observe between precision and recall?
**Answer:**
"In retrieval, precision and recall exhibited a classic inverse relationship with respect to $top\_k$. At $top\_k=1$, precision reached ~0.83 but recall hovered at ~0.47. As $top\_k$ grew to 5, recall approached 1.00 for all queries, but precision dropped down to ~0.42. For narrow factual queries, a low $top\_k$ (1–2) was optimal because a single chunk contained the entire answer. For multi-part queries (such as detailing full-stack technologies across frontend, backend, and database), a higher $top\_k$ (3–4) was mandatory to cross the recall threshold."
