"""Sequential, resumable 375-generation RAG evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import os,sys,json,csv,time,math,re,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.config import get_settings
from app.aws.bedrock_service import BedrockEmbeddingService
from app.aws.llm_service import BedrockLLMService
from app.models.chunk import DocumentChunk
from app.services.pdf_extractor import PDFExtractor
from app.services.text_chunker import TextChunker
from app.rag.vector_store import VectorStoreService
OUT=ROOT/'evaluation_results'; PLOTS=OUT/'plots'; DOC_DIR=ROOT/'data/rag_eval/full_matrix_docs'
TEMPS=[0.0,0.05,0.1,0.2,0.3]; TOPKS=[1,2,3,4,5]
DOCS=[(1,'synthetic_insurance_policy_1_page.pdf',80101),(2,'synthetic_insurance_policy_2_pages.pdf',80102),(5,'synthetic_insurance_policy_5_pages.pdf',80105),(6,'synthetic_insurance_policy_6_pages.pdf',80106),(7,'synthetic_insurance_policy_7_pages.pdf',80107)]
QS=[
 {'id':'Q01','type':'short_factual','q':'What is the policy number?','ref':'The policy number is CSM-HO-2048-TEST.','facts':['policy_number=CSM-HO-2048-TEST'],'answerable':True},
 {'id':'Q13','type':'long_multi_part','q':'Summarize the insured, policy period, dwelling limit, two deductible rules, and one major exclusion.','ref':'The named insured is Taylor Reed; the policy runs January 1, 2026 to January 1, 2027; dwelling coverage is $350,000; the standard deductible is $1,500; the wind/hail deductible is 2% of dwelling coverage ($7,000) and replaces the standard deductible for those losses; flood/surface water is excluded.','facts':['insured=Taylor Reed','period=2026-01-01_to_2027-01-01','dwelling=$350,000','standard_deductible=$1,500','wind_hail_deductible=$7,000','wind_hail_replaces_standard=true','flood_excluded=true'],'answerable':True},
 {'id':'Q11','type':'unanswerable','q':'What exact percentage improvement in claims approval accuracy did Cedar Shield achieve after deploying its AI system?','ref':'The document does not provide a claims-approval accuracy improvement percentage.','facts':[],'answerable':False}]
SYS='You are an insurance document assistant. Answer only from the supplied context. Do not invent policy details. If the answer cannot be determined from the context, clearly state that the document does not provide enough information. Do not make unsupported assumptions. Do not expose internal reasoning or chain-of-thought. Provide a direct, clear, professional answer citing page numbers when relevant.'
IN_RATE=.30; OUT_RATE=2.50; EMB_RATE=.02; DELAY=max(0,float(os.getenv('RAG_EVAL_DELAY_SECONDS','.25'))); RETRIES=3
EMBED_EVENTS=[]

def utc(): return datetime.now(timezone.utc).isoformat()
def retrying_embedder(service):
 original=service.embed_text
 def call(text):
  for attempt in range(1,RETRIES+2):
   try:
    result=original(text);EMBED_EVENTS.append({'timestamp_utc':utc(),'status':'success','attempt':attempt,'input_chars':len(text)})
    if DELAY:time.sleep(DELAY)
    return result
   except Exception as ex:
    msg=f'{type(ex).__name__}: {str(ex)[:300]}';transient=any(x in (type(ex).__name__+' '+str(ex)).lower() for x in ['throttl','serviceunavailable','modelnotready','internalserver','timeout','temporarily unavailable'])
    EMBED_EVENTS.append({'timestamp_utc':utc(),'status':'retryable_failure' if transient and attempt<=RETRIES else 'failure','attempt':attempt,'input_chars':len(text),'error':msg})
    if transient and attempt<=RETRIES:
     time.sleep(min(8,.5*2**(attempt-1)));continue
    raise
 service.embed_text=call
 return service
def savej(p,x):
 # On Windows the destination can be held briefly by antivirus/indexing after
 # each checkpoint. Use a unique same-directory temp and bounded retries.
 payload=json.dumps(x,indent=2,ensure_ascii=False)
 t=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
 t.write_text(payload,encoding='utf8')
 for attempt in range(7):
  try:
   os.replace(t,p); return
  except PermissionError:
   if attempt==6: raise
   time.sleep(min(2.0,.2*(2**attempt)))
def canon(s): return re.sub(r'[^a-z0-9]+',' ',s.lower())
def supports(f,a):
 x=canon(a); d=re.sub(r'[^0-9.]','',a)
 if f.startswith('policy_number'): return 'csm ho 2048 test' in x
 if f.startswith('insured='): return 'taylor reed' in x
 if f.startswith('period='): return bool(re.search(r'jan(?:uary)?\s+1,?\s+2026',a,re.I) and re.search(r'jan(?:uary)?\s+1,?\s+2027',a,re.I))
 if f.startswith('dwelling='): return '350000' in d
 if f.startswith('standard_deductible='): return '1500' in d
 if f.startswith('wind_hail_deductible='): return '2%' in a and '7000' in d and ('wind' in x or 'hail' in x)
 if f.startswith('wind_hail_replaces'): return any(z in x for z in ['replaces the standard','replace the standard','not added','not add','instead of the standard'])
 if f.startswith('flood_excluded'): return 'flood' in x and any(z in x for z in ['excluded','not covered','exclusion'])
 return False
def abstains(a): return any(x in a.lower() for x in ['does not provide','does not state','does not specify','not provided','not specified','cannot be determined','insufficient information','no information'])
def unsupported_nums(a,ctx):
 found=[]; context=re.sub(r'[^0-9.]','',ctx)
 for pat in [r'\$\s*[\d,]+(?:\.\d+)?',r'\b\d+(?:\.\d+)?\s*%',r'\b\d+\s+(?:calendar|business)\s+days?\b']:
  for m in re.finditer(pat,a,re.I):
   n=re.sub(r'[^0-9.]','',m.group())
   if n and n not in context and m.group() not in found and not re.search(r'page\s*$',a[max(0,m.start()-10):m.start()],re.I): found.append(m.group())
 return found
def score(q,a,chunks):
 ctx='\n'.join(c['text'] for c in chunks)
 if not q['answerable']:
  ab=abstains(a); bad=bool(re.search(r'\b\d+(?:\.\d+)?\s*%',a) or re.search(r'achieved|improved by',a,re.I)) and not ab
  return {'supported_facts':[],'missing_facts':[],'unsupported_claims':['unsupported improvement asserted'] if bad else [],'answer_precision':None,'answer_recall':None,'answer_f1':None,'correctness':int(ab and not bad),'completeness':None,'faithfulness':1.0 if ab and not bad else 0.0,'groundedness':1.0 if ab and not bad else 0.0,'hallucination_rate':float(bad),'abstention_accuracy':int(ab),'evaluation_notes':'Unanswerable: F1 N/A; abstention scored separately; percentage detector is limited.'}
 fs=q['facts']; good=[f for f in fs if supports(f,a)]; miss=[f for f in fs if f not in good]; bad=unsupported_nums(a,ctx); den=len(good)+len(bad); p=len(good)/den if den else 0; r=len(good)/len(fs) if fs else None; f1=2*p*r/(p+r) if r is not None and p+r else 0
 faith=sum(supports(f,ctx) for f in good)/den if den else 0
 return {'supported_facts':good,'missing_facts':miss,'unsupported_claims':bad,'answer_precision':p,'answer_recall':r,'answer_f1':f1,'correctness':int(not miss and not bad),'completeness':r,'faithfulness':faith,'groundedness':faith,'hallucination_rate':len(bad)/den if den else 0,'abstention_accuracy':None,'evaluation_notes':'Atomic fact matcher; numeric/date unsupported-claim detector is lexical and incomplete.'}
def ret_metrics(ids,rel,k,n):
 if not rel: return {'retrieval_precision':None,'retrieval_recall':None,'retrieval_f1':None,'mrr':None,'ndcg':None,'context_relevance':None,'evidence_coverage':None}
 r=ids[:k]; hits=[x for x in r if x in rel]; p=len(hits)/min(k,n); rec=len(set(hits))/len(set(rel)); f=2*p*rec/(p+rec) if p+rec else 0
 rr=next((1/(i+1) for i,x in enumerate(r) if x in rel),0)
 dcg=sum(1/math.log2(i+2) for i,x in enumerate(r) if x in rel); ideal=sum(1/math.log2(i+2) for i in range(min(k,len(set(rel)))))
 return {'retrieval_precision':p,'retrieval_recall':rec,'retrieval_f1':f,'mrr':rr,'ndcg':dcg/ideal if ideal else 0,'context_relevance':p,'evidence_coverage':rec}
def cited(a): return [int(x) for x in re.findall(r'\bpage\s*(\d+)',a,re.I)]
def citations(a,chunks,q):
 c=cited(a)
 if not q['answerable']: return {'citation_precision':None,'citation_coverage':None,'cited_pages':c}
 sp={x['page_number'] for x in chunks}; ev={x['page_number'] for x in chunks if any(supports(f,x['text']) for f in q['facts'])}
 return {'citation_precision':sum(x in sp for x in c)/len(c) if c else 0,'citation_coverage':len(set(c)&ev)/len(ev) if ev else 0,'cited_pages':c}
def dumpcsv(path,rows,cols=None):
 if cols is None: cols=list(rows[0]) if rows else ['status']
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
def make_agg(rows):
 output=[]
 metrics=['retrieval_precision','retrieval_recall','retrieval_f1','mrr','ndcg','answer_precision','answer_recall','answer_f1','correctness','completeness','faithfulness','groundedness','hallucination_rate','abstention_accuracy','retrieval_latency_ms','generation_latency_ms','end_to_end_latency_ms','input_tokens','output_tokens','estimated_generation_cost_usd']
 groups={}
 for row in rows: groups.setdefault((row['temperature'],row['retrieval_top_k']),[]).append(row)
 for (t,k),g in groups.items():
  x={'temperature':t,'retrieval_top_k':k,'n_primary_planned':15,'n_primary_completed':len(g)}
  for m in metrics:
   vals=[float(r[m]) for r in g if r.get(m) not in (None,'')]
   if vals:x[m+'_mean']=sum(vals)/len(vals)
  x['cost_usd_sum']=sum(float(r.get('estimated_generation_cost_usd') or 0) for r in g);output.append(x)
 return output
def persist(state):
 savej(OUT/'ALL_ANSWERS.json',state)
 matrix=[]; metrics=[]
 for e in state['matrix']:
  matrix.append({k:e.get(k) for k in ['pair_id','document','document_pages','question_id','question_type','temperature','retrieval_top_k','status','attempt_count','retrieved_chunk_ids','retrieval_metrics','retrieval_latency_ms','generation_latency_ms','end_to_end_latency_ms','input_tokens','output_tokens','estimated_generation_cost_usd','failure_or_retry_notes']})
  if e.get('answer_metrics'):
   metrics.append({k:e.get(k) for k in ['pair_id','document','document_pages','question_id','question_type','question','temperature','retrieval_top_k']}|e['retrieval_metrics']|e['answer_metrics']|{k:e.get(k) for k in ['retrieval_latency_ms','generation_latency_ms','end_to_end_latency_ms','input_tokens','output_tokens','estimated_generation_cost_usd','status']})
 dumpcsv(OUT/'EXECUTION_MATRIX.csv',matrix)
 dumpcsv(OUT/'PER_QUESTION_METRICS.csv',metrics)
 dumpcsv(OUT/'AGGREGATE_CONFIGURATION_METRICS.csv',make_agg([r for r in metrics if r.get('status')=='success']))
 state['completed_primary']=sum(x['status']=='success' for x in state['matrix']);state['failed_primary']=sum(x['status']=='failed' for x in state['matrix']);state['updated_at_utc']=utc()
 state['run_status']='completed' if state['completed_primary']==375 else ('partial' if state.get('stopped_reason') else 'running')
 savej(OUT/'ALL_ANSWERS.json',state)
def retrieve_relevant(q,chunks):
 return [c.get('chunk_id',c.get('chunk_index')) for c in chunks if q['answerable'] and any(supports(f,c.get('text',c.get('chunk_text',''))) for f in q['facts'])]
def audit_files(audit,temps):
 lines=['# Ground-truth audit','',f"Temperature matrix follows the user-requested values `{temps}`. Bundle manifest values 0.23 and 0.4 are excluded.",'','Three questions per document: Q01 short factual; Q13 long cross-section; Q11 unanswerable. Q01 and every Q13 atomic fact were checked against extracted text. Q11 is supported by explicit statements that claims-approval accuracy/real claim rates are not provided.','']
 for d in audit:
  lines += [f"## {d['filename']} — {d['pages']} pages",f"Extracted characters: {d['chars']}; chunk count: {len(d['chunks'])}; chunk_size={d['chunk_size']}; overlap={d['chunk_overlap']}",f"SHA-256: `{d['sha256']}`","Chunk ID / page / Q01 relevant / Q13 relevant:"]
  lines += [f"- `{c['chunk_id']}` / {c['page_number']} / {c['q01']} / {c['q13']}" for c in d['chunks']];lines.append('')
 lines += ['Relevance mapping is deterministic: Q01 chunks contain the policy number; Q13 chunks contain at least one verified atomic fact. Q11 has no relevant answer chunk; retrieval relevance metrics are N/A.']
 (OUT/'GROUND_TRUTH_AUDIT.md').write_text('\n'.join(lines),encoding='utf8')
def preestimate():
 # Prior successful pipeline smoke used 786 input / 33 output at top_k=3.
 inp=375*786;lo=375*100;hi=375*200;emb=.0015
 expected=(inp*IN_RATE+lo*OUT_RATE)/1e6+emb; upper=(inp*IN_RATE+hi*OUT_RATE)/1e6+emb
 # conservative text bound: 1125 retrieved chunks x 999 chars, 250 prompt tokens/call, 2048 output tokens/call
 hardin=1125*999+375*250;hardout=375*2048
 hard=(hardin*IN_RATE+hardout*OUT_RATE)/1e6+emb
 return {'expected_usd_range':[round(expected,4),round(upper,4)],'hard_primary_bound_usd':round(hard,4),'four_attempt_stress_usd':round(hard*4,4),'pricing_usd_per_million':{'Nova input':IN_RATE,'Nova output':OUT_RATE,'Titan input':EMB_RATE},'expected_input_tokens':inp,'expected_output_tokens_range':[lo,hi],'note':'Public-rate estimate using earlier 786/33 token smoke baseline; embedding estimate approximate; max bound assumes 1 token/character plus prompt overhead and full 2048 output tokens.'}
def prepare(settings,vs):
 ext=PDFExtractor(); chkr=TextChunker(); docs=[]; audit=[]
 for pages,name,docid in DOCS:
  raw=(DOC_DIR/name).read_bytes(); pdatas=ext.extract_pages(raw)
  if len(pdatas)!=pages: raise RuntimeError(f'Page count mismatch {name}: {len(pdatas)}')
  chunks=chkr.chunk_document_pages(pdatas); alltext=' '.join(x['text'] for x in pdatas).lower()
  q01='csm-ho-2048-test' in alltext
  q13={'insured':'taylor reed' in alltext,'period':'2026' in alltext and '2027' in alltext,'dwelling':'350,000' in alltext,'standard deductible':'1,500' in alltext,'wind percentage':'2%' in alltext,'wind amount':'7,000' in alltext,'flood exclusion':'flood' in alltext and ('excluded' in alltext or 'not covered' in alltext)}
  q11=('claims approval rate' in alltext or 'claims-approval accuracy' in alltext or 'claim approval rate' in alltext) and ('not provided' in alltext or 'does not provide' in alltext or 'does not state' in alltext)
  if not q01 or not all(q13.values()) or not q11: raise RuntimeError(f'Ground truth check failed: {name}, {q01}, {q13}, {q11}')
  models=[DocumentChunk(document_id=docid,chunk_index=c['chunk_index'],page_number=c['page_number'],chunk_text=c['chunk_text'],file_name=name,document_type='Synthetic Home Insurance Test Policy') for c in chunks]
  old=vs._collection.get(where={'document_id':docid},include=['metadatas'])
  if len(old.get('ids',[]))<len(models): vs.add_chunks(models)
  gold=[]
  for c in chunks: gold.append({'chunk_id':c['chunk_index'],'page_number':c['page_number'],'q01':'csm-ho-2048-test' in c['chunk_text'].lower(),'q13':any(supports(f,c['chunk_text']) for f in QS[1]['facts'])})
  d={'pages':pages,'filename':name,'doc_id':docid,'chunks':chunks,'page_data':pdatas,'sha256':hashlib.sha256(raw).hexdigest()};docs.append(d)
  audit.append({'filename':name,'pages':pages,'chars':sum(len(x['text']) for x in pdatas),'chunks':gold,'chunk_size':settings.chunk_size,'chunk_overlap':settings.chunk_overlap,'sha256':d['sha256'],'q01_verified':q01,'q13_verified_facts':q13,'q11_verified_unanswerable':q11})
 audit_files(audit,TEMPS);return docs,audit
def initial_state(base,estimate):
 matrix=[]
 for pages,name,_ in DOCS:
  for q in QS:
   for t in TEMPS:
    for k in TOPKS:matrix.append({'pair_id':f'{pages}p_{q["id"]}','document':name,'document_pages':pages,'question_id':q['id'],'question_type':q['type'],'question':q['q'],'reference_answer':q['ref'],'expected_facts':q['facts'],'answerable':q['answerable'],'temperature':t,'retrieval_top_k':k,'status':'pending','attempt_count':0,'attempt_log':[]})
 if len(matrix)!=375:raise RuntimeError(f'Expected 375 matrix entries, got {len(matrix)}')
 return {'experiment_id':'rag_full_matrix_375_v1','created_at_utc':utc(),'config':{'temperatures':TEMPS,'retrieval_top_k':TOPKS,'model_id':base.bedrock_model_id,'region':base.aws_region,'embedding_model_id':base.bedrock_embedding_model_id,'embedding_dimension':base.embedding_dimension,'chunk_size':base.chunk_size,'chunk_overlap':base.chunk_overlap,'max_output_tokens':base.bedrock_max_tokens,'delay_seconds':DELAY,'max_retries':RETRIES,'primary_generations_planned':375,'repeat_generations':0,'pricing_usd_per_million':{'input':IN_RATE,'output':OUT_RATE,'embedding':EMB_RATE}},'preflight_cost_estimate':estimate,'document_audit':[],'smoke_tests':[],'matrix':matrix,'completed_primary':0,'failed_primary':0,'run_status':'running','stopped_reason':None}
def main():
 OUT.mkdir(exist_ok=True);PLOTS.mkdir(exist_ok=True)
 base=get_settings();settings=base.model_copy(update={'chroma_persist_directory':'./data/rag_eval/full_matrix_chromadb','chroma_collection_name':'rag_full_matrix_eval'})
 estimate=preestimate()
 report=f'''# Full-Matrix RAG Evaluation\n\n**Status: RUNNING.** 375 primary generations are planned; checkpoint progress is in `ALL_ANSWERS.json`.\n\n## Pre-execution workload and estimate\nFive documents × three questions × five temperatures × five top_k values = 375 primary generations. Temperatures: `{TEMPS}`; top_k: `{TOPKS}`. Repeats: 0. Retries counted separately. Expected token-cost range: ${estimate['expected_usd_range'][0]:.4f}–${estimate['expected_usd_range'][1]:.4f}; conservative full-output bound: ${estimate['hard_primary_bound_usd']:.4f}; four-attempt full-charge stress bound: ${estimate['four_attempt_stress_usd']:.4f}. Based on Nova 2 Lite $0.30/M input and $2.50/M output; Titan V2 $0.02/M. Public rates: [Nova 2 Lite](https://aws.amazon.com/blogs/machine-learning/object-detection-with-amazon-nova-2-lite/) and [Titan V2](https://aws.amazon.com/blogs/machine-learning/get-started-with-amazon-titan-text-embeddings-v2-a-new-state-of-the-art-embeddings-model-on-amazon-bedrock/). Estimate uses prior smoke input of 786 tokens at k=3, expected 100–200 output tokens/request; it is not a billing quote.\n\nConfigured model `{base.bedrock_model_id}` Region `{base.aws_region}`; embedding `{base.bedrock_embedding_model_id}`; chunking {base.chunk_size}/{base.chunk_overlap}; max output {base.bedrock_max_tokens}. Sequential execution, configurable {DELAY}s delay, up to 3 retries with bounded exponential backoff. No judge calls.\n\n## Scoring formulas\nRetrieval P@k = relevant retrieved / min(k, document chunks); recall = relevant retrieved / total labeled relevant chunks; F1 harmonic mean; MRR reciprocal rank of first relevant; nDCG binary relevance; context relevance=P@k; evidence coverage=recall. Q11 retrieval metrics are N/A because there are no relevant answer chunks.\n\nFor Q01/Q13, answer recall = supported expected facts / all expected facts. Answer precision = supported expected facts / (supported facts + detected unsupported numeric/date claims); F1 is harmonic mean. Correctness is binary all facts present and no detected unsupported numeric/date claims; completeness=recall. Faithfulness and groundedness are deterministic evidence-overlap proxies. The numeric/date detector is limited, so hallucination rate means detected unsupported numeric/date claims divided by detected claims, not exhaustive hallucination adjudication. For Q11, answer precision/recall/F1/completeness are N/A; abstention accuracy is separate. Citation precision = valid cited retrieved pages / all cited pages; citation coverage = evidence-bearing retrieved pages cited / evidence-bearing retrieved pages. Retrieval latency includes query embedding plus Chroma; generation latency uses Bedrock metadata or wall fallback; end-to-end is their sum. Estimated cost uses token usage and public rates. Aggregates are arithmetic means over completed rows.\n\n'''
 (OUT/'FINAL_RAG_EVALUATION_REPORT.md').write_text(report,encoding='utf8')
 (OUT/'REPRODUCTION_INSTRUCTIONS.md').write_text(f'''Run from repository root: `python scripts/run_full_matrix_rag_evaluation.py`. Input PDFs are in `data/rag_eval/full_matrix_docs`. The runner uses the app PDFExtractor, TextChunker, Titan embedding service, isolated persistent Chroma collection, and configured BedrockLLMService. Matrix: temperatures {TEMPS}, top_k {TOPKS}; 375 primary rows. No 0.23 or 0.4. Calls are sequential; set `RAG_EVAL_DELAY_SECONDS` (default {DELAY}). Each request has max 3 retries after the first attempt (bounded exponential backoff, max 8 seconds). Each attempt/result is checkpointed; successful keys are skipped after restart. Model `{base.bedrock_model_id}`, Region `{base.aws_region}`. Answerable metrics use deterministic atomic-fact rules; Q11 answer F1 is N/A. No LLM judges. Public rates: Nova ${IN_RATE}/M input, ${OUT_RATE}/M output; Titan ${EMB_RATE}/M input.\n''',encoding='utf8')
 (OUT/'STAR_INTERVIEW_ANSWER.md').write_text('''# STAR answer — RAG evaluation\n\n**Situation:** Five fictional insurance policies from one to seven pages were used to evaluate an existing RAG application.\n\n**Task:** Compare five temperatures and five retrieval depths using the same short factual, multi-part, and unanswerable question types for each document.\n\n**Action:** The evaluation uses the app's PDF extraction, chunking, Titan V2 embeddings, Chroma retrieval, and Nova 2 Lite generation. Requests run sequentially with bounded retries and checkpoints. Answerable questions are scored against verified atomic facts; unanswerable questions are scored for abstention separately. No LLM judge calls are used.\n\n**Result:** This answer is updated after the run. It reports only completed calls and measured outcomes; it does not claim a result while the matrix is incomplete.\n''',encoding='utf8')
 cp=OUT/'ALL_ANSWERS.json';existing=None
 if cp.exists():
  try:
   candidate=json.loads(cp.read_text(encoding='utf8'))
   if candidate.get('experiment_id')=='rag_full_matrix_375_v1':existing=candidate
  except Exception:pass
 state=existing or initial_state(base,estimate);persist(state)
 # Explicit, verifiable manifest workload; reuse only this experiment's own checkpoint.
 embedder=retrying_embedder(BedrockEmbeddingService(settings=settings));vs=VectorStoreService(embedding_service=embedder,settings=settings)
 docs,audit=prepare(settings,vs)
 state['embedding_events']=EMBED_EVENTS
 state['document_audit']=[{k:v for k,v in d.items() if k not in ('chunks','page_data')} for d in docs];persist(state)
 cp=OUT/'ALL_ANSWERS.json';state=None
 if cp.exists():
  try:
   old=json.loads(cp.read_text(encoding='utf8'))
   if old.get('experiment_id')=='rag_full_matrix_375_v1':state=old
  except Exception:pass
 if state is None:
  matrix=[]
  for pages,name,_ in DOCS:
   for q in QS:
    for t in TEMPS:
     for k in TOPKS:matrix.append({'pair_id':f'{pages}p_{q["id"]}','document':name,'document_pages':pages,'question_id':q['id'],'question_type':q['type'],'question':q['q'],'reference_answer':q['ref'],'expected_facts':q['facts'],'answerable':q['answerable'],'temperature':t,'retrieval_top_k':k,'status':'pending','attempt_count':0,'attempt_log':[]})
  if len(matrix)!=375:raise RuntimeError(f'Expected 375 matrix entries, got {len(matrix)}')
  state={'experiment_id':'rag_full_matrix_375_v1','created_at_utc':utc(),'config':{'temperatures':TEMPS,'retrieval_top_k':TOPKS,'model_id':base.bedrock_model_id,'region':base.aws_region,'embedding_model_id':base.bedrock_embedding_model_id,'embedding_dimension':base.embedding_dimension,'chunk_size':base.chunk_size,'chunk_overlap':base.chunk_overlap,'max_output_tokens':base.bedrock_max_tokens,'delay_seconds':DELAY,'max_retries':RETRIES,'primary_generations_planned':375,'repeat_generations':0,'pricing_usd_per_million':{'input':IN_RATE,'output':OUT_RATE,'embedding':EMB_RATE}},'preflight_cost_estimate':estimate,'document_audit':[{k:v for k,v in d.items() if k not in ('chunks','page_data')} for d in docs],'smoke_tests':[],'matrix':matrix,'completed_primary':0,'failed_primary':0,'run_status':'running','stopped_reason':None}
 smoke_path=OUT/'full_matrix_smoke_test.json'
 if smoke_path.exists() and not state.get('smoke_tests'):
  state['smoke_tests']=[json.loads(smoke_path.read_text(encoding='utf8'))]
 state['run_status']='running';persist(state)
 # Cache retrieval per question/document/top_k so temperature comparisons share exact ranked contexts.
 retrieval={}
 for d in docs:
  for q in QS:
   emb_start=time.perf_counter();query_embedding=vs.embedding_service.embed_text(q['q']);embedding_ms=(time.perf_counter()-emb_start)*1000
   for k in TOPKS:
    start=time.perf_counter();raw=vs._collection.query(query_embeddings=[query_embedding],n_results=min(k,len(d['chunks'])),where={'document_id':d['doc_id']},include=['documents','metadatas','distances']);lat=embedding_ms+(time.perf_counter()-start)*1000
    found=[]
    for i,vector_id in enumerate(raw.get('ids',[[]])[0]):
     meta=raw['metadatas'][0][i];dist=raw['distances'][0][i]
     found.append({'vector_id':vector_id,'chunk_id':meta.get('chunk_id'),'chunk_index':meta.get('chunk_index'),'page_number':meta.get('page_number'),'text':raw['documents'][0][i],'similarity_score':round(max(0,min(1,1-dist)),4)})
    # Keep stable numeric chunk IDs and relevance labels.
    for c in found:c['relevant']=any(supports(f,c['text']) for f in q['facts']) if q['answerable'] else False
    rel=retrieve_relevant(q,d['chunks']);retrieval[(d['pages'],q['id'],k)]={'chunks':found,'latency':lat,'metrics':ret_metrics([c['chunk_id'] for c in found],rel,k,len(d['chunks'])),'relevant_ids':rel}
 state['embedding_events']=EMBED_EVENTS;persist(state)
 llm=BedrockLLMService(settings=base); lookup={(e['document_pages'],e['question_id'],e['temperature'],e['retrieval_top_k']):e for e in state['matrix']}
 for pages,name,docid in DOCS:
  d=next(x for x in docs if x['doc_id']==docid)
  for q in QS:
   for t in TEMPS:
    for k in TOPKS:
     e=lookup[(pages,q['id'],t,k)]
     if e['status']=='success':continue
     if e['status']=='failed':raise RuntimeError(f"Checkpoint has an exhausted failed row: {pages}p {q['id']} T={t} k={k}: {e.get('failure_or_retry_notes')}")
     r=retrieval[(pages,q['id'],k)];chunks=r['chunks']
     context='\n\n'.join(f"[Excerpt {i+1} | Page {c['page_number']} | Chunk ID: {c['chunk_id']}]\n{c['text']}" for i,c in enumerate(chunks))
     prompt=f"Document Name: {name}\nDocument Category: Synthetic Home Insurance Test Policy\n\nRetrieved Document Context:\n{context}\n\nUser Question: {q['q']}\n\nAnswer:"
     e.update({'status':'in_progress','retrieved_chunk_ids':[c['chunk_id'] for c in chunks],'retrieved_chunks':chunks,'labeled_relevant_chunk_ids':r['relevant_ids'],'retrieval_metrics':r['metrics'],'retrieval_latency_ms':r['latency'],'prompt':prompt});persist(state)
     first_attempt=e.get('attempt_count',0)+1
     for attempt in range(first_attempt,RETRIES+2):
      e['attempt_count']+=1; start=time.perf_counter()
      try:
       res=llm.generate_text_with_metadata(prompt=prompt,system_prompt=SYS,temperature=t);wall=(time.perf_counter()-start)*1000;lat=res.get('latency_ms') or wall
       inp=int(res.get('input_tokens',0));out=int(res.get('output_tokens',0));cost=(inp*IN_RATE+out*OUT_RATE)/1e6
       am=score(q,res['text'],chunks)|citations(res['text'],chunks,q)
       e.update({'status':'success','generated_answer':res['text'],'generation_latency_ms':lat,'generation_wall_latency_ms':wall,'end_to_end_latency_ms':lat+r['latency'],'input_tokens':inp,'output_tokens':out,'total_tokens':inp+out,'estimated_generation_cost_usd':cost,'answer_metrics':am,'failure_or_retry_notes':''})
       e['attempt_log'].append({'attempt':attempt,'status':'success','latency_ms':lat,'input_tokens':inp,'output_tokens':out,'timestamp_utc':utc()});persist(state)
       print(f"{state['completed_primary']}/375 success; {pages}p {q['id']} T={t} k={k}, {lat:.0f}ms, {inp}/{out} tokens",flush=True);time.sleep(DELAY);break
      except Exception as ex:
       msg=f'{type(ex).__name__}: {str(ex)[:400]}';s=(type(ex).__name__+' '+str(ex)).lower();transient=any(z in s for z in ['throttl','serviceunavailable','modelnotready','internalserver','timeout','temporarily unavailable'])
       e['attempt_log'].append({'attempt':attempt,'status':'retryable_failure' if transient and attempt<=RETRIES else 'failure','error':msg,'timestamp_utc':utc()});e['failure_or_retry_notes']=msg;persist(state)
       if transient and attempt<=RETRIES:
        delay=min(8,.5*2**(attempt-1));print(f'transient failure {attempt}/{RETRIES}; retry in {delay}s',flush=True);time.sleep(delay);continue
       e['status']='failed';state['stopped_reason']=msg;persist(state);raise RuntimeError(f'Stopping at {pages}p {q["id"]}, T={t}, k={k}: {msg}') from ex
 state['run_status']='completed';state['stopped_reason']=None;persist(state);finish(state)
 print('Completed all 375 primary generations.')
def mean_by(rows,key,metric,answerable=False):
 groups={}
 for r in rows:
  if r.get('status')!='success' or (answerable and r.get('question_type')=='unanswerable'):continue
  try:v=float(r.get(metric))
  except (TypeError,ValueError):continue
  groups.setdefault(str(r.get(key)),[]).append(v)
 return [(k,sum(v)/len(v)) for k,v in groups.items()]
def svg_bars(filename,title,items):
 if not items:return
 w,h=1000,560; left,right,top,bottom=90,30,70,115; pw=w-left-right; ph=h-top-bottom
 vmax=max([v for _,v in items] or [1]) or 1; gap=pw/len(items); bw=min(75,gap*.62)
 p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><text x="{w/2}" y="35" text-anchor="middle" font-family="Arial" font-size="22">{title}</text>']
 for i,(lab,val) in enumerate(items):
  x=left+i*gap+(gap-bw)/2; bh=ph*max(0,val)/vmax; y=top+ph-bh
  p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="#3973ac"/><text x="{x+bw/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{val:.3f}</text><text x="{x+bw/2:.1f}" y="{top+ph+24}" text-anchor="middle" font-family="Arial" font-size="12">{lab}</text>')
 p.append(f'<path d="M{left},{top} V{top+ph} H{w-right}" fill="none" stroke="#333"/></svg>');(PLOTS/filename).write_text(''.join(p),encoding='utf8')
def svg_heat(rows):
 if not rows:return
 ts=sorted({float(r['temperature']) for r in rows});ks=sorted({int(float(r['retrieval_top_k'])) for r in rows});lookup={(float(r['temperature']),int(float(r['retrieval_top_k']))):float(r.get('answer_f1_mean') or 0) for r in rows}
 x0,y0,cw,ch=160,100,85,55;w,h=700,460;p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><text x="350" y="35" text-anchor="middle" font-family="Arial" font-size="20">Answer F1 heatmap</text>']
 for j,k in enumerate(ks):p.append(f'<text x="{x0+j*cw+cw/2}" y="{y0-12}" text-anchor="middle">{k}</text>')
 for i,t in enumerate(ts):
  p.append(f'<text x="{x0-20}" y="{y0+i*ch+ch/2+5}" text-anchor="end">{t}</text>')
  for j,k in enumerate(ks):
   v=lookup.get((t,k),0);c=int(240-180*v);p.append(f'<rect x="{x0+j*cw}" y="{y0+i*ch}" width="{cw}" height="{ch}" fill="#{c:02x}{c:02x}ee" stroke="white"/><text x="{x0+j*cw+cw/2}" y="{y0+i*ch+ch/2+5}" text-anchor="middle">{v:.2f}</text>')
 p.append('</svg>');(PLOTS/'answer_f1_heatmap.svg').write_text(''.join(p),encoding='utf8')
def finish(state):
 rows=list(csv.DictReader((OUT/'PER_QUESTION_METRICS.csv').open(encoding='utf-8-sig')));ok=[r for r in rows if r.get('status')=='success']; answerable=[r for r in ok if r.get('question_type')!='unanswerable']
 PLOTS.mkdir(exist_ok=True)
 for m,title,f in [('correctness','Temperature vs correctness','temperature_vs_correctness.svg'),('answer_f1','Temperature vs answer F1','temperature_vs_answer_f1.svg'),('faithfulness','Temperature vs faithfulness proxy','temperature_vs_faithfulness.svg')]:svg_bars(f,title,mean_by(answerable,'temperature',m))
 for m,title,f in [('retrieval_precision','top_k vs retrieval precision','topk_precision.svg'),('retrieval_recall','top_k vs retrieval recall','topk_recall.svg'),('retrieval_f1','top_k vs retrieval F1','topk_f1.svg')]:svg_bars(f,title,mean_by(answerable,'retrieval_top_k',m))
 svg_bars('hallucination_rate.svg','Detected hallucination proxy',mean_by(ok,'question_type','hallucination_rate'))
 svg_bars('abstention_accuracy.svg','Abstention accuracy',mean_by(ok,'question_type','abstention_accuracy'))
 svg_bars('document_length_answer_f1.svg','Document length vs answer F1',mean_by(answerable,'document_pages','answer_f1'))
 svg_bars('question_type_answer_f1.svg','Question type vs answer F1',mean_by(answerable,'question_type','answer_f1'))
 ag=make_agg(answerable);svg_heat(ag)
 for metric,title,fn in [('generation_latency_ms_mean','Mean generation latency by temperature/top_k','latency_by_configuration.svg'),('estimated_generation_cost_usd_mean','Mean per-call cost by temperature/top_k','cost_by_configuration.svg')]:svg_bars(fn,title,[(f"{r['temperature']}/{r['retrieval_top_k']}",float(r[metric])) for r in ag if r.get(metric) not in ('',None)])
 ag.sort(key=lambda r:(-float(r.get('answer_f1_mean') or 0),-float(r.get('faithfulness_mean') or 0),-float(r.get('correctness_mean') or 0),float(r.get('generation_latency_ms_mean') or 1e12)))
 best='Unavailable (no completed answerable scores).'
 if ag:
  b=ag[0];best=f"T={b['temperature']}, top_k={b['retrieval_top_k']}; answer F1={float(b.get('answer_f1_mean') or 0):.3f}, faithfulness proxy={float(b.get('faithfulness_mean') or 0):.3f}, correctness={float(b.get('correctness_mean') or 0):.3f}, latency={float(b.get('generation_latency_ms_mean') or 0):.1f} ms."
 retries=sum(max(0,x['attempt_count']-1) for x in state['matrix']);cost=sum(float(r.get('estimated_generation_cost_usd') or 0) for r in ok)
 p=OUT/'FINAL_RAG_EVALUATION_REPORT.md';text=p.read_text(encoding='utf8');text=text.replace('**Status: PRE-FLIGHT — primary matrix not yet started.**',f"**Status: {state['run_status'].upper()} — {state['completed_primary']}/375 primary generations succeeded; {state['failed_primary']} failed.** Retries: {retries}; repeats: 0.")
 text+=f"\n## Observed results\n\nSuccessful calls: {state['completed_primary']}/375; failed: {state['failed_primary']}; retries: {retries}; repeats: 0. Estimated generation cost from reported token usage: USD {cost:.6f}. Best overall answerable configuration by mean F1, then faithfulness/correctness/latency: {best}\n\nCharts are in plots/. Unanswerable answer F1 is N/A. Hallucination rate is a limited lexical proxy, not exhaustive adjudication.\n";p.write_text(text,encoding='utf8')
 p=OUT/'STAR_INTERVIEW_ANSWER.md';p.write_text(p.read_text(encoding='utf8')+f"\nMeasured: {state['completed_primary']}/375 primary calls; estimated generation cost USD {cost:.6f}; retries {retries}. See final report for quality results and limits.\n",encoding='utf8')
if __name__=='__main__':
 try:main()
 except Exception as ex:
  cp=OUT/'ALL_ANSWERS.json'
  if cp.exists():
   try:
    state=json.loads(cp.read_text(encoding='utf8'))
    if state.get('experiment_id')=='rag_full_matrix_375_v1':
     state['run_status']='partial';state['stopped_reason']=f'{type(ex).__name__}: {str(ex)[:500]}';persist(state);finish(state)
   except Exception as checkpoint_error:print(f'Could not finalize partial report: {checkpoint_error}',file=sys.stderr)
  raise
