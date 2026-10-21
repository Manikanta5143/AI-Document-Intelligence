# Full-Matrix RAG Evaluation

**Status: COMPLETED — 375/375 primary generations succeeded; 0 failed.** One additional successful model generation was repeated after a checkpoint write error; total model invocations: 376.

## Pre-execution workload and estimate
Five documents × three questions × five temperatures × five top_k values = 375 primary generations. Temperatures: `[0.0, 0.05, 0.1, 0.2, 0.3]`; top_k: `[1, 2, 3, 4, 5]`. Repeats: 0. Retries counted separately. Expected token-cost range: $0.1837–$0.2774; conservative full-output bound: $2.2868; four-attempt full-charge stress bound: $9.1471. Based on Nova 2 Lite $0.30/M input and $2.50/M output; Titan V2 $0.02/M. Public rates: [Nova 2 Lite rates](https://aws.amazon.com/blogs/machine-learning/build-an-ai-powered-equipment-repair-assistant-using-amazon-bedrock-agentcore/) and [Titan V2](https://aws.amazon.com/blogs/machine-learning/get-started-with-amazon-titan-text-embeddings-v2-a-new-state-of-the-art-embeddings-model-on-amazon-bedrock/). Estimate uses prior smoke input of 786 tokens at k=3, expected 100–200 output tokens/request; it is not a billing quote.

Configured model `global.amazon.nova-2-lite-v1:0` Region `ap-south-1`; embedding `amazon.titan-embed-text-v2:0`; chunking 1000/200; max output 2048. Sequential execution, configurable 0.25s delay, up to 3 retries with bounded exponential backoff. No judge calls.

## Scoring formulas
Retrieval P@k = relevant retrieved / min(k, document chunks); recall = relevant retrieved / total labeled relevant chunks; F1 harmonic mean; MRR reciprocal rank of first relevant; nDCG binary relevance; context relevance=P@k; evidence coverage=recall. Q11 retrieval metrics are N/A because there are no relevant answer chunks.

For Q01/Q13, answer recall = supported expected facts / all expected facts. Answer precision = supported expected facts / (supported facts + detected unsupported numeric/date claims); F1 is harmonic mean. Correctness is binary all facts present and no detected unsupported numeric/date claims; completeness=recall. Faithfulness and groundedness are deterministic evidence-overlap proxies. The numeric/date detector is limited, so hallucination rate means detected unsupported numeric/date claims divided by detected claims, not exhaustive hallucination adjudication. For Q11, answer precision/recall/F1/completeness are N/A; abstention accuracy is separate. Citation precision = valid cited retrieved pages / all cited pages; citation coverage = evidence-bearing retrieved pages cited / evidence-bearing retrieved pages. Retrieval latency includes query embedding plus Chroma; generation latency uses Bedrock metadata or wall fallback; end-to-end is their sum. Estimated cost uses token usage and public rates. Aggregates are arithmetic means over completed rows.


## Observed results

- **Primary matrix:** 375/375 successful; 0 failed. Model invocations: 376, including 1 replay after a checkpoint write failed after an earlier generation had succeeded. The replay recovered that answer; the original success and checkpoint error are retained in its `attempt_log`. No Bedrock throttling retries were observed.
- **Generation cost estimate:** USD 0.189539 from recorded token usage for all 376 successful invocations (USD 0.188807 across primary matrix rows plus USD 0.000732 for the replay). Uses public rates and excludes Titan embedding charges because embedding token usage was not recorded. This is an estimate, not an AWS invoice.
- **Quota and budget:** The configured account/profile, Region, and model were inspected. Exact current Bedrock quotas, AWS credit balance, and billing budget could not be retrieved because the configured identity lacked access to the required quota, Cost Explorer, and Budgets reads. No credentials are included in this report. The completed run itself had no throttling, but remaining quota/credits cannot be asserted.
- **Smoke test:** One-page Q01 at T=0, top_k=3 succeeded before the matrix: 786 input / 33 output tokens, 807 ms, estimated USD 0.0003183. See `quota_conscious_smoke_test.json`. Its one-page PDF SHA-256 (`9313c174…`) differs from the current bundle one-page PDF (`50e3e625…`), so it validates the configured Bedrock path and same Q01 but not the exact current file; the first primary matrix row then exercised the current file end-to-end.

### Results by document length

Means over 50 answerable primary rows per document (10 question/config rows × 5 temperatures; Q11 excluded).

| Document | n | Retrieval P | Retrieval R | Retrieval F1 | MRR | nDCG | Answer P | Answer R | Answer F1 | Correctness | Completeness | Faithfulness proxy | Groundedness proxy | Detected hallucination rate | Gen latency ms | Cost/call USD |
|---|---|---|---|---|---|---|---|---|---|
| 1 page | 50 | 0.650 | 0.950 | 0.723 | 1.000 | 1.000 | 1.000 | 0.929 | 0.944 | 0.900 | 0.929 | 1.000 | 1.000 | 0.000 | 875.420 | 0.000428 |
| 2 page | 50 | 0.708 | 0.783 | 0.688 | 1.000 | 0.913 | 1.000 | 0.903 | 0.940 | 0.680 | 0.903 | 0.992 | 0.992 | 0.000 | 983.200 | 0.000564 |
| 5 page | 50 | 0.628 | 0.550 | 0.411 | 0.700 | 0.752 | 0.900 | 0.683 | 0.746 | 0.400 | 0.683 | 0.900 | 0.900 | 0.000 | 1199.620 | 0.000610 |
| 6 page | 50 | 0.628 | 0.536 | 0.396 | 0.700 | 0.752 | 0.900 | 0.683 | 0.745 | 0.400 | 0.683 | 0.900 | 0.900 | 0.000 | 1061.740 | 0.000593 |
| 7 page | 50 | 0.545 | 0.336 | 0.279 | 0.550 | 0.586 | 0.700 | 0.477 | 0.541 | 0.200 | 0.477 | 0.700 | 0.700 | 0.000 | 1084.240 | 0.000590 |

### Results by question category

| Question type | n | Retrieval P | Retrieval R | Retrieval F1 | MRR | nDCG | Answer P | Answer R | Answer F1 | Correctness | Completeness | Faithfulness proxy | Groundedness proxy | Detected hallucination rate | Abstention accuracy | Gen latency ms | Cost/call USD |
|---|---|---|---|---|---|---|---|---|---|---|
| short_factual | 125 | 0.363 | 0.780 | 0.456 | 0.580 | 0.636 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.000 | N/A | 714.568 | 0.000333 |
| long_multi_part | 125 | 0.901 | 0.482 | 0.543 | 1.000 | 0.965 | 1.000 | 0.670 | 0.767 | 0.232 | 0.670 | 0.997 | 0.997 | 0.000 | N/A | 1367.120 | 0.000781 |
| unanswerable | 125 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 1.000 | N/A | 1.000 | 1.000 | 0.000 | 1.000 | 780.832 | 0.000396 |

For unanswerable questions, answer precision/recall/F1 and completeness are N/A by design; abstention accuracy is reported separately. In this deterministic check, 125/125 unanswerable responses abstained.

### Question category by document length

Each answerable cell averages 25 runs (5 temperatures × 5 top_k). Unanswerable cells report abstention rather than answer F1.

| Document | Question type | n | Answer F1 | Correctness | Completeness | Faithfulness proxy | Abstention accuracy |
|---|---|---|---|---|---|---|---|
| 1 page | short_factual | 25 | 1.000 | 1.000 | 1.000 | 1.000 | N/A |
| 1 page | long_multi_part | 25 | 0.889 | 0.800 | 0.857 | 1.000 | N/A |
| 1 page | unanswerable | 25 | N/A | 1.000 | N/A | 1.000 | 1.000 |
| 2 page | short_factual | 25 | 1.000 | 1.000 | 1.000 | 1.000 | N/A |
| 2 page | long_multi_part | 25 | 0.881 | 0.360 | 0.806 | 0.984 | N/A |
| 2 page | unanswerable | 25 | N/A | 1.000 | N/A | 1.000 | 1.000 |
| 5 page | short_factual | 25 | 0.800 | 0.800 | 0.800 | 0.800 | N/A |
| 5 page | long_multi_part | 25 | 0.691 | 0.000 | 0.566 | 1.000 | N/A |
| 5 page | unanswerable | 25 | N/A | 1.000 | N/A | 1.000 | 1.000 |
| 6 page | short_factual | 25 | 0.800 | 0.800 | 0.800 | 0.800 | N/A |
| 6 page | long_multi_part | 25 | 0.690 | 0.000 | 0.566 | 1.000 | N/A |
| 6 page | unanswerable | 25 | N/A | 1.000 | N/A | 1.000 | 1.000 |
| 7 page | short_factual | 25 | 0.400 | 0.400 | 0.400 | 0.400 | N/A |
| 7 page | long_multi_part | 25 | 0.682 | 0.000 | 0.554 | 1.000 | N/A |
| 7 page | unanswerable | 25 | N/A | 1.000 | N/A | 1.000 | 1.000 |

### Temperature and retrieval-depth comparisons

| Temperature | Answer F1 | Correctness | Faithfulness proxy | Generation latency ms |
|---|---|---|---|---|
| 0 | 0.781 | 0.520 | 0.900 | 1018.020 |
| 0.05 | 0.787 | 0.520 | 0.896 | 1070.080 |
| 0.1 | 0.785 | 0.520 | 0.900 | 1019.520 |
| 0.2 | 0.783 | 0.520 | 0.900 | 1051.740 |
| 0.3 | 0.781 | 0.500 | 0.896 | 1044.860 |

| top_k | Retrieval P | Retrieval R | Retrieval F1 | Answer F1 | Correctness | Generation latency ms |
|---|---|---|---|---|---|---|
| 1 | 0.700 | 0.262 | 0.335 | 0.396 | 0.200 | 1048.960 |
| 2 | 0.700 | 0.590 | 0.535 | 0.786 | 0.500 | 1131.600 |
| 3 | 0.600 | 0.651 | 0.509 | 0.838 | 0.500 | 966.220 |
| 4 | 0.600 | 0.813 | 0.566 | 0.948 | 0.700 | 1019.580 |
| 5 | 0.560 | 0.841 | 0.554 | 0.948 | 0.680 | 1037.860 |

The strongest aggregate answer F1 in the 25-configuration CSV is at T=0.1, top_k=5 (0.950); treat that as a descriptive winner on this small repeated-question set, not a general optimum. Top_k=4 had similar mean answer F1 (0.948) at lower per-call input than top_k=5.

### Interpretation and measurement limits

Answer quality declines with longer documents in this test: mean answer F1 was 0.944 for the one-page document, 0.941 for two pages, 0.746 for five pages, 0.745 for six pages, and 0.541 for seven pages. These summaries are specific to the synthetic policies, one repeated question per type, and this chunking/retrieval setup. Long multi-part correctness is strict (all seven expected facts required), so correctness (0.232) is lower than its mean fact recall (0.670).

The reported hallucination rate is 0.000 **for the implemented limited lexical detector only**; this does not establish zero hallucinations. It flags selected unsupported numeric/date claims and can miss unsupported qualitative statements. Faithfulness/groundedness are deterministic fact-overlap proxies, not human or LLM judgments. No extra LLM judge calls were made. Retrieval relevance is based on fixed lexical atomic-fact labels; Q11 relevance metrics are N/A. Latency excludes client-side time outside the recorded retrieval and generation timings.

Charts generated for this run are in `plots/`: answer F1 by temperature/top_k, retrieval metrics by top_k, temperature/correctness/faithfulness, document length, question type, latency, cost, detected hallucination proxy, and abstention accuracy. The SVG charts are primary; any older PNG files in this folder may belong to earlier experiments.

### Next workload recommendation

The exact remaining budget and quota are unavailable, so do not authorize additional calls on an assumed balance. Once those reads are available, prefer adding a second independently verified question in each category for each of the five documents: that adds 15 question pairs × 25 configurations = 375 calls, for 750 total (not 480). At the observed mean primary-generation cost, that extension is approximately USD 0.189 before embedding charges. A 480-call total would add only 105 calls and would not complete that full question expansion. Temperature means varied little (answer F1 0.781–0.787), while top_k results varied substantially (0.396 at k=1 vs 0.948 at k=4); if the budget only supports a smaller follow-up, targeted independent repeats at k=1, 4, and 5 would help assess stability around retrieval depth. No further calls were run.
