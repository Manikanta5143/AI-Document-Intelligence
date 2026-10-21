"""Finalize the report with verified matrix-wide and subgroup results."""
from pathlib import Path
import csv, json, re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation_results'
def read_csv(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
def mean(rows,key):
    vals=[float(r[key]) for r in rows if r.get(key) not in ('',None)]
    return sum(vals)/len(vals) if vals else None
def fmt(x): return 'N/A' if x is None else f'{x:.3f}'
def fmt_cost(rows):
    x=mean(rows,'estimated_generation_cost_usd')
    return 'N/A' if x is None else f'{x:.6f}'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(x) for x in r)+' |' for r in rows)

state=json.loads((OUT/'ALL_ANSWERS.json').read_text(encoding='utf-8'))
metrics=read_csv('PER_QUESTION_METRICS.csv')
ok=[r for r in metrics if r.get('status')=='success']
answerable=[r for r in ok if r['question_type']!='unanswerable']
attempt_successes=[(r,a) for r in state['matrix'] for a in r.get('attempt_log',[]) if a.get('status')=='success']
actual_calls=len(attempt_successes)
primary_cost=sum(float(r.get('estimated_generation_cost_usd') or 0) for r in ok)
unrecorded_cost=0.0
for row,attempt in attempt_successes:
    # Successes beyond one per matrix row represent billed replays not reflected in the primary metrics CSV.
    successful_logs=[a for a in row.get('attempt_log',[]) if a.get('status')=='success']
    if attempt is successful_logs[0] and len(successful_logs)>1:
        # The persisted primary response is the last success; earlier success was replayed after a checkpoint error.
        unrecorded_cost += (int(attempt.get('input_tokens',0))*.30+int(attempt.get('output_tokens',0))*2.50)/1_000_000
        break
total_cost=primary_cost+unrecorded_cost
state['actual_model_generation_invocations']=actual_calls
state['successful_primary_generations']=len(ok)
state['actual_replay_generations']=actual_calls-len(ok)
state['estimated_recorded_primary_generation_cost_usd']=round(primary_cost,9)
state['estimated_unrecorded_replay_cost_usd']=round(unrecorded_cost,9)
state['estimated_total_generation_cost_usd']=round(total_cost,9)
state['quota_and_budget_read_status']='Exact current Bedrock quota, AWS credits, and billing budget could not be read; preflight access denied. See final report.'
(OUT/'ALL_ANSWERS.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8')

docrows=[]
for pages in [1,2,5,6,7]:
    g=[r for r in answerable if int(r['document_pages'])==pages]
    docrows.append([f'{pages} page',len(g),fmt(mean(g,'retrieval_precision')),fmt(mean(g,'retrieval_recall')),fmt(mean(g,'retrieval_f1')),fmt(mean(g,'mrr')),fmt(mean(g,'ndcg')),fmt(mean(g,'answer_precision')),fmt(mean(g,'answer_recall')),fmt(mean(g,'answer_f1')),fmt(mean(g,'correctness')),fmt(mean(g,'completeness')),fmt(mean(g,'faithfulness')),fmt(mean(g,'groundedness')),fmt(mean(g,'hallucination_rate')),fmt(mean(g,'generation_latency_ms')),fmt_cost(g)])
typerows=[]
for typ in ['short_factual','long_multi_part','unanswerable']:
    g=[r for r in ok if r['question_type']==typ]
    typerows.append([typ,len(g),fmt(mean(g,'retrieval_precision')),fmt(mean(g,'retrieval_recall')),fmt(mean(g,'retrieval_f1')),fmt(mean(g,'mrr')),fmt(mean(g,'ndcg')),fmt(mean(g,'answer_precision')),fmt(mean(g,'answer_recall')),fmt(mean(g,'answer_f1')),fmt(mean(g,'correctness')),fmt(mean(g,'completeness')),fmt(mean(g,'faithfulness')),fmt(mean(g,'groundedness')),fmt(mean(g,'hallucination_rate')),fmt(mean(g,'abstention_accuracy')),fmt(mean(g,'generation_latency_ms')),fmt_cost(g)])
cross=[]
for pages in [1,2,5,6,7]:
    for typ in ['short_factual','long_multi_part','unanswerable']:
        g=[r for r in ok if int(r['document_pages'])==pages and r['question_type']==typ]
        cross.append([f'{pages} page',typ,len(g),fmt(mean(g,'answer_f1')),fmt(mean(g,'correctness')),fmt(mean(g,'completeness')),fmt(mean(g,'faithfulness')),fmt(mean(g,'abstention_accuracy'))])
temp=[]
for t in [0,.05,.1,.2,.3]:
    g=[r for r in answerable if float(r['temperature'])==t]
    temp.append([t,fmt(mean(g,'answer_f1')),fmt(mean(g,'correctness')),fmt(mean(g,'faithfulness')),fmt(mean(g,'generation_latency_ms'))])
top=[]
for k in [1,2,3,4,5]:
    g=[r for r in answerable if int(float(r['retrieval_top_k']))==k]
    top.append([k,fmt(mean(g,'retrieval_precision')),fmt(mean(g,'retrieval_recall')),fmt(mean(g,'retrieval_f1')),fmt(mean(g,'answer_f1')),fmt(mean(g,'correctness')),fmt(mean(g,'generation_latency_ms'))])

report=OUT/'FINAL_RAG_EVALUATION_REPORT.md'
text=report.read_text(encoding='utf-8')
text=re.sub(r'\*\*Status:.*?\*\*.*?(?=\n\n## Pre-execution)',f"**Status: COMPLETED — {len(ok)}/375 primary generations succeeded; {len(state.get('failed_primary',[])) if isinstance(state.get('failed_primary'),list) else state.get('failed_primary',0)} failed.** One additional successful model generation was repeated after a checkpoint write error; total model invocations: {actual_calls}.",text,count=1,flags=re.S)
text=re.sub(r'\n## Observed results\n.*$', '', text, flags=re.S)
text += f'''\n## Observed results\n\n- **Primary matrix:** {len(ok)}/375 successful; 0 failed. Model invocations: {actual_calls}, including {actual_calls-len(ok)} replay after a checkpoint write failed after an earlier generation had succeeded. The replay recovered that answer; the original success and checkpoint error are retained in its `attempt_log`. No Bedrock throttling retries were observed.\n- **Generation cost estimate:** USD {total_cost:.6f} from recorded token usage for all {actual_calls} successful invocations (USD {primary_cost:.6f} across primary matrix rows plus USD {unrecorded_cost:.6f} for the replay). Uses public rates and excludes Titan embedding charges because embedding token usage was not recorded. This is an estimate, not an AWS invoice.\n- **Quota and budget:** The configured account/profile, Region, and model were inspected. Exact current Bedrock quotas, AWS credit balance, and billing budget could not be retrieved because the configured identity lacked access to the required quota, Cost Explorer, and Budgets reads. No credentials are included in this report. The completed run itself had no throttling, but remaining quota/credits cannot be asserted.\n- **Smoke test:** One-page Q01 at T=0, top_k=3 succeeded before the matrix: 786 input / 33 output tokens, 807 ms, estimated USD 0.0003183. See `quota_conscious_smoke_test.json`.\n\n### Results by document length\n\nMeans over 50 answerable primary rows per document (10 question/config rows × 5 temperatures; Q11 excluded).\n\n{table(['Document','n','Retrieval P','Retrieval R','Retrieval F1','Answer F1','Correctness','Completeness','Faithfulness proxy','Gen latency ms'],docrows)}\n\n### Results by question category\n\n{table(['Question type','n','Retrieval P','Retrieval R','Answer P','Answer R','Answer F1','Correctness','Faithfulness proxy','Abstention accuracy','Gen latency ms'],typerows)}\n\nFor unanswerable questions, answer precision/recall/F1 and completeness are N/A by design; abstention accuracy is reported separately. In this deterministic check, 75/75 unanswerable responses abstained.\n\n### Question category by document length\n\nEach answerable cell averages 25 runs (5 temperatures × 5 top_k). Unanswerable cells report abstention rather than answer F1.\n\n{table(['Document','Question type','n','Answer F1','Correctness','Completeness','Faithfulness proxy','Abstention accuracy'],cross)}\n\n### Temperature and retrieval-depth comparisons\n\n{table(['Temperature','Answer F1','Correctness','Faithfulness proxy','Generation latency ms'],temp)}\n\n{table(['top_k','Retrieval P','Retrieval R','Retrieval F1','Answer F1','Correctness','Generation latency ms'],top)}\n\nThe strongest aggregate answer F1 in the 25-configuration CSV is at T=0.1, top_k=5 (0.950); treat that as a descriptive winner on this small repeated-question set, not a general optimum. Top_k=4 had similar mean answer F1 (0.948) at lower per-call input than top_k=5.\n\n### Interpretation and measurement limits\n\nAnswer quality declines with longer documents in this test: mean answer F1 was 0.944 for the one-page document, 0.941 for two pages, 0.746 for five pages, 0.745 for six pages, and 0.541 for seven pages. These summaries are specific to the synthetic policies, one repeated question per type, and this chunking/retrieval setup. Long multi-part correctness is strict (all seven expected facts required), so correctness (0.232) is lower than its mean fact recall (0.670).\n\nThe reported hallucination rate is 0.000 **for the implemented limited lexical detector only**; this does not establish zero hallucinations. It flags selected unsupported numeric/date claims and can miss unsupported qualitative statements. Faithfulness/groundedness are deterministic fact-overlap proxies, not human or LLM judgments. No extra LLM judge calls were made. Retrieval relevance is based on fixed lexical atomic-fact labels; Q11 relevance metrics are N/A. Latency excludes client-side time outside the recorded retrieval and generation timings.\n\nCharts generated for this run are in `plots/`: answer F1 by temperature/top_k, retrieval metrics by top_k, temperature/correctness/faithfulness, document length, question type, latency, cost, detected hallucination proxy, and abstention accuracy. The SVG charts are primary; any older PNG files in this folder may belong to earlier experiments.\n'''
text=text.replace("| Document | n | Retrieval P | Retrieval R | Retrieval F1 | Answer F1 | Correctness | Completeness | Faithfulness proxy | Gen latency ms |", "| Document | n | Retrieval P | Retrieval R | Retrieval F1 | MRR | nDCG | Answer P | Answer R | Answer F1 | Correctness | Completeness | Faithfulness proxy | Groundedness proxy | Detected hallucination rate | Gen latency ms | Cost/call USD |")
text=text.replace("| Question type | n | Retrieval P | Retrieval R | Answer P | Answer R | Answer F1 | Correctness | Faithfulness proxy | Abstention accuracy | Gen latency ms |", "| Question type | n | Retrieval P | Retrieval R | Retrieval F1 | MRR | nDCG | Answer P | Answer R | Answer F1 | Correctness | Completeness | Faithfulness proxy | Groundedness proxy | Detected hallucination rate | Abstention accuracy | Gen latency ms | Cost/call USD |")
text=text.replace('75/75 unanswerable responses abstained','125/125 unanswerable responses abstained')
text=text.replace('[Nova 2 Lite](https://aws.amazon.com/blogs/machine-learning/object-detection-with-amazon-nova-2-lite/)', '[Nova 2 Lite rates](https://aws.amazon.com/blogs/machine-learning/build-an-ai-powered-equipment-repair-assistant-using-amazon-bedrock-agentcore/)')
text=text.replace('See `quota_conscious_smoke_test.json`.','See `quota_conscious_smoke_test.json`. Its one-page PDF SHA-256 (`9313c174…`) differs from the current bundle one-page PDF (`50e3e625…`), so it validates the configured Bedrock path and same Q01 but not the exact current file; the first primary matrix row then exercised the current file end-to-end.')
text += '''\n### Next workload recommendation\n\nThe exact remaining budget and quota are unavailable, so do not authorize additional calls on an assumed balance. Once those reads are available, prefer adding a second independently verified question in each category for each of the five documents: that adds 15 question pairs × 25 configurations = 375 calls, for 750 total (not 480). At the observed mean primary-generation cost, that extension is approximately USD 0.189 before embedding charges. A 480-call total would add only 105 calls and would not complete that full question expansion. Temperature means varied little (answer F1 0.781–0.787), while top_k results varied substantially (0.396 at k=1 vs 0.948 at k=4); if the budget only supports a smaller follow-up, targeted independent repeats at k=1, 4, and 5 would help assess stability around retrieval depth. No further calls were run.\n'''
report.write_text(text,encoding='utf-8')

star=OUT/'STAR_INTERVIEW_ANSWER.md'
star.write_text(f'''# STAR answer — RAG evaluation\n\n**Situation:** Five synthetic insurance policies (1, 2, 5, 6, and 7 pages) were used to evaluate the existing RAG application across short factual, multi-part, and unanswerable questions.\n\n**Task:** Complete the quota-conscious 375-primary-generation matrix across five temperatures and five retrieval depths without running the larger workload or adding LLM judges.\n\n**Action:** Used the app's PDF extraction, 1000/200 chunking, Titan V2 embeddings, Chroma retrieval, and Nova 2 Lite in `ap-south-1`. The run was sequential with checkpointing and bounded retries. I verified the question facts against extracted PDF text, scored answers deterministically, and retained token usage, retrieved chunk IDs, latency, failures, and attempt logs.\n\n**Result:** Completed all 375 primary rows with no Bedrock throttling. One generation was repeated after Windows denied a checkpoint replacement, producing 376 successful model invocations in total. Primary-row generation cost is estimated at USD {primary_cost:.6f}; including the replay it is USD {total_cost:.6f}, excluding unmeasured embedding charges. Aggregate answer F1 was highest at T=0.1/top_k=5 (0.950); mean answer F1 fell from 0.944 on the one-page policy to 0.541 on the seven-page policy. This is a deterministic small-set evaluation, not proof of zero hallucinations or a universal parameter optimum. Exact remaining quota, credits, and billing budget were not accessible.\n''',encoding='utf-8')
print(f'Finalized report; {len(ok)} primary successes, {actual_calls} actual model calls, estimated generation cost ${total_cost:.6f}')
