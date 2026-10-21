# STAR answer — RAG evaluation

**Situation:** Five synthetic insurance policies (1, 2, 5, 6, and 7 pages) were used to evaluate the existing RAG application across short factual, multi-part, and unanswerable questions.

**Task:** Complete the quota-conscious 375-primary-generation matrix across five temperatures and five retrieval depths without running the larger workload or adding LLM judges.

**Action:** Used the app's PDF extraction, 1000/200 chunking, Titan V2 embeddings, Chroma retrieval, and Nova 2 Lite in `ap-south-1`. The run was sequential with checkpointing and bounded retries. I verified the question facts against extracted PDF text, scored answers deterministically, and retained token usage, retrieved chunk IDs, latency, failures, and attempt logs.

**Result:** Completed all 375 primary rows with no Bedrock throttling. One generation was repeated after Windows denied a checkpoint replacement, producing 376 successful model invocations in total. Primary-row generation cost is estimated at USD 0.188807; including the replay it is USD 0.189539, excluding unmeasured embedding charges. Aggregate answer F1 was highest at T=0.1/top_k=5 (0.950); mean answer F1 fell from 0.944 on the one-page policy to 0.541 on the seven-page policy. This is a deterministic small-set evaluation, not proof of zero hallucinations or a universal parameter optimum. Exact remaining quota, credits, and billing budget were not accessible.
