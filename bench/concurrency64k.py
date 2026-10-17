"""Exact 64K input / 1K output, one and two simultaneous requests, two rounds each."""
import argparse,ast,hashlib,json,random,statistics,subprocess,threading,time,urllib.request
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoTokenizer
from sglang.srt.entrypoints.openai.encoding_dsv41 import encode_messages

p=argparse.ArgumentParser()
p.add_argument('--model-path',required=True)
p.add_argument('--port',type=int,default=39504)
p.add_argument('--out',required=True)
p.add_argument('--rounds',type=int,default=2)
p.add_argument('--input-tokens',type=int,default=65536)
p.add_argument('--output-tokens',type=int,default=1024)
args=p.parse_args()
tok=AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
A=SimpleNamespace(exact=True)
tree=ast.parse(Path(__file__).with_name('speed_points.py').read_text())
nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name=='document') or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TOPICS' for t in n.targets))]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<benchmark-document>','exec'))
base=f'http://127.0.0.1:{args.port}'
results={'input_tokens':args.input_tokens,'output_tokens':args.output_tokens,'temperature':0,'ignore_eos':True,'rounds':args.rounds,'rows':[]}
def inputs(seed):
 text,code=document(args.input_tokens,seed)
 ids=tok.encode(encode_messages([{'role':'user','content':text}],thinking_mode='chat'),add_special_tokens=False)
 cut=len(ids)//2
 if len(ids)>args.input_tokens:ids=ids[:cut]+ids[cut+len(ids)-args.input_tokens:]
 elif len(ids)<args.input_tokens:ids=ids[:cut]+[tok.encode('\n',add_special_tokens=False)[0]]*(args.input_tokens-len(ids))+ids[cut:]
 assert len(ids)==args.input_tokens
 return ids,code
for concurrency in [1,2]:
 for rnd in range(args.rounds):
  prepared=[inputs(20261018+concurrency*100+rnd*10+i) for i in range(concurrency)]
  barrier=threading.Barrier(concurrency+1); rows=[None]*concurrency
  def run(i):
   ids,code=prepared[i]
   payload=json.dumps({'input_ids':ids,'stream':True,'sampling_params':{'temperature':0,'max_new_tokens':args.output_tokens,'ignore_eos':True}}).encode()
   request=urllib.request.Request(base+'/generate',payload,{'Content-Type':'application/json'})
   barrier.wait();start=time.monotonic();first=last=None;first_count=last_count=0;meta={};answer='';error=None
   try:
    with urllib.request.urlopen(request,timeout=1200) as response:
     for line in response:
      if not line.startswith(b'data:'):continue
      data=line[5:].strip()
      if data==b'[DONE]':continue
      obj=json.loads(data);meta=obj.get('meta_info') or meta
      count=meta.get('completion_tokens',0)
      if count and count!=last_count:
       now=time.monotonic()
       if first is None:first=now;first_count=count
       last=now;last_count=count
      answer=obj.get('text',answer)
   except Exception as exc:error=repr(exc)
   rows[i]={'request':i,'input_tokens':len(ids),'output_tokens':last_count,'start':start,'first':first,'last':last,'first_count':first_count,
            'ttft_s':first-start if first else None,'prefill_tok_s':len(ids)/(first-start) if first else None,
            'decode_tok_s':(last_count-first_count)/(last-first) if first and last>first else None,
            'spec_accept_length':meta.get('spec_accept_length'),'spec_verify_ct':meta.get('spec_verify_ct'),
            'cached_tokens':meta.get('cached_tokens'),'recall':code in answer,'error':error,
            'output_sha256':hashlib.sha256(answer.encode()).hexdigest()}
  workers=[threading.Thread(target=run,args=(i,)) for i in range(concurrency)]
  for w in workers:w.start()
  barrier.wait()
  for w in workers:w.join()
  assert all(r and not r['error'] and r['recall'] and r['output_tokens']==args.output_tokens for r in rows),rows
  window=max(r['last'] for r in rows)-min(r['first'] for r in rows)
  wall=max(r['last'] for r in rows)-min(r['start'] for r in rows)
  overlap=min(r['last'] for r in rows)-max(r['first'] for r in rows)
  record={'concurrency':concurrency,'round':rnd,'requests':rows,
          'per_request_decode_tok_s':statistics.mean(r['decode_tok_s'] for r in rows),
          'prefill_tok_s_mean':statistics.mean(r['prefill_tok_s'] for r in rows),
          'ttft_s_mean':statistics.mean(r['ttft_s'] for r in rows),
          'aggregate_decode_tok_s':sum(r['output_tokens'] for r in rows)/window,
          'aggregate_end_to_end_tok_s':sum(r['output_tokens'] for r in rows)/wall,
          'decode_overlap_seconds':overlap if concurrency>1 else None}
  if concurrency>1:assert overlap>0,record
  results['rows'].append(record)
  Path(args.out).write_text(json.dumps(results,indent=2))
  print(json.dumps(record),flush=True)
results['summary']={}
for c in [1,2]:
 rs=[r for r in results['rows'] if r['concurrency']==c]
 results['summary'][str(c)]={key:statistics.mean(r[key] for r in rs) for key in ['per_request_decode_tok_s','prefill_tok_s_mean','ttft_s_mean','aggregate_decode_tok_s','aggregate_end_to_end_tok_s']}
Path(args.out).write_text(json.dumps(results,indent=2))
print('CONCURRENCY_64K1K_PASSED',json.dumps(results['summary']),flush=True)
