"""Which matrix products does DeepSeek V4.1 run, with which shapes and dtypes?

Usage: python ds41_gemm_shapes.py [--port 39503]
Profiles one short prefill (eager, so CPU ops carry shapes) with record_shapes and prints
matmul-like aten ops grouped by (op, input dims, input types) with call counts, plus the
GPU kernels each group launched (via correlation ids).
"""
import argparse, collections, glob, gzip, json, os, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=39503)
P.add_argument('--out', default='/tmp/ds41-gemm-shapes')
A = P.parse_args()
BASE = f'http://127.0.0.1:{A.port}'
os.makedirs(A.out, exist_ok=True)
for f in glob.glob(os.path.join(A.out, '*')):
    os.remove(f)
def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=1800).read()
post('/start_profile', dict(output_dir=A.out, num_steps=1, activities=['CPU', 'GPU'], record_shapes=True,
                            profile_prefix='shapes'))
post('/generate', dict(text='Summarise: ' + 'the quick brown fox jumps over the lazy dog. ' * 60,
                       sampling_params=dict(temperature=0, max_new_tokens=1)))
deadline = time.time() + 300
while not glob.glob(os.path.join(A.out, '*.trace.json*')) and time.time() < deadline:
    time.sleep(2)
time.sleep(5)
path = sorted(glob.glob(os.path.join(A.out, '*.trace.json*')))[0]
ev = json.load(gzip.open(path, 'rt'))['traceEvents']
ops = [e for e in ev if e.get('ph') == 'X' and e.get('cat') == 'cpu_op' and
       any(k in e['name'] for k in ('mm', 'linear', 'matmul', 'einsum', 'bmm', 'gemm'))]
rt = {e['args']['correlation']: e for e in ev if e.get('cat') == 'cuda_runtime' and 'correlation' in e.get('args', {})}
kernels = collections.defaultdict(list)
for e in ev:
    if e.get('cat') == 'kernel' and 'correlation' in e.get('args', {}):
        kernels[e['args']['correlation']].append(e)
# attach each runtime launch to the innermost enclosing cpu op on the same thread
by_tid = collections.defaultdict(list)
for o in ops:
    by_tid[o['tid']].append(o)
groups = collections.defaultdict(lambda: dict(n=0, kern=collections.Counter(), us=0))
for o in ops:
    key = (o['name'], json.dumps(o['args'].get('Input Dims')), json.dumps(o['args'].get('Input type')))
    groups[key]['n'] += 1
for c, r in rt.items():
    cands = [o for o in by_tid.get(r['tid'], []) if o['ts'] <= r['ts'] <= o['ts'] + o['dur']]
    if not cands:
        continue
    o = min(cands, key=lambda o: o['dur'])
    key = (o['name'], json.dumps(o['args'].get('Input Dims')), json.dumps(o['args'].get('Input type')))
    for k in kernels.get(c, []):
        groups[key]['kern'][k['name'][:70]] += 1; groups[key]['us'] += k['dur']
for key, g in sorted(groups.items(), key=lambda kv: -kv[1]['n'])[:40]:
    print(g['n'], key[0], key[1], key[2], round(g['us']), dict(g['kern'].most_common(2)))
