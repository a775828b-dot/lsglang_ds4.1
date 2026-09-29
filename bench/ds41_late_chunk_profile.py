"""Profile a late 32K prefill chunk of a ~500K-token prompt (where prefill slows down).

Usage: python ds41_late_chunk_profile.py [--port 39503] [--delay 230] [--out DIR]
Starts a long prompt, waits --delay seconds (chunks near 400K context), profiles one
scheduler iteration, and prints busy time per kernel (top 25) and per family.
"""
import argparse, collections, glob, gzip, json, os, random, threading, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=39503)
P.add_argument('--delay', type=float, default=230)
P.add_argument('--out', default='/tmp/ds41-late-chunk')
A = P.parse_args()
BASE = f'http://127.0.0.1:{A.port}'
os.makedirs(A.out, exist_ok=True)
for f in glob.glob(os.path.join(A.out, '*')):
    os.remove(f)


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout).read()


rng = random.Random(time.time_ns())
text = ' '.join(f'entry {i}: value {rng.randint(0, 10**6)} for item {rng.randint(0, 10**4)}.' for i in range(int(os.environ.get("ENTRIES", "28000"))))
th = threading.Thread(target=lambda: post('/generate', dict(text=text, sampling_params=dict(max_new_tokens=1))), daemon=True)
th.start()
time.sleep(A.delay)
post('/start_profile', dict(output_dir=A.out, num_steps=1, activities=['CPU', 'GPU'] if os.environ.get('WITH_STACK') else ['GPU'],
                            with_stack=bool(os.environ.get('WITH_STACK')), profile_prefix='late'))
th.join()
deadline = time.time() + 300
while not glob.glob(os.path.join(A.out, '*.trace.json*')) and time.time() < deadline:
    time.sleep(1)
time.sleep(3)
ev = json.load(gzip.open(sorted(glob.glob(os.path.join(A.out, '*.trace.json*')))[0], 'rt'))['traceEvents']
g = [e for e in ev if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
t0 = min(e['ts'] for e in g); t1 = max(e['ts'] + e['dur'] for e in g)
top = collections.Counter(); cnt = collections.Counter()
for e in g:
    top[e['name'][:95]] += e['dur']; cnt[e['name'][:95]] += 1
print(json.dumps(dict(span_s=round((t1 - t0) / 1e6, 2), kernel_s=round(sum(top.values()) / 1e6, 2))))
for k, v in top.most_common(25):
    print(f'{v / 1e3:9.1f} ms  x{cnt[k]:5d}  {k}')

if os.environ.get('WITH_STACK'):
    # attribute kernels to the innermost Python function (model code) that launched them
    rt = {e['args']['correlation']: e for e in ev if e.get('cat') == 'cuda_runtime' and 'correlation' in e.get('args', {})}
    py = collections.defaultdict(list)
    for e in ev:
        if e.get('ph') == 'X' and e.get('cat') == 'python_function' and 'sglang' in e['name'] and '/torch/' not in e['name']:
            py[e['tid']].append(e)
    agg = collections.Counter(); calls = collections.Counter()
    for e in g:
        r = rt.get(e.get('args', {}).get('correlation'))
        if not r:
            continue
        enc = [p for p in py.get(r['tid'], []) if p['ts'] <= r['ts'] <= p['ts'] + p['dur']]
        if not enc:
            continue
        name = min(enc, key=lambda p: p['dur'])['name']
        agg[name] += e['dur']; calls[name] += 1
    print('--- kernel time by innermost sglang function')
    for k, v in agg.most_common(20):
        print(f'{v / 1e3:9.1f} ms  x{calls[k]:6d}  {k[:120]}')
