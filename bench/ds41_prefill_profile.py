"""Where does one DeepSeek V4.1 prefill go? GPU trace of a single fresh prompt.

Usage: python ds41_prefill_profile.py [--port 39503] [--words 1400] [--out DIR]
Profiles the scheduler iteration that runs the prefill (GPU activity) and prints the
main-stream span, busy time, the idle gaps by size, and busy time by kernel family.
"""
import argparse, collections, glob, gzip, json, os, random, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=39503)
P.add_argument('--words', type=int, default=1400)
P.add_argument('--out', default='/tmp/ds41-prefill-profile')
A = P.parse_args()
BASE = f'http://127.0.0.1:{A.port}'
os.makedirs(A.out, exist_ok=True)
for f in glob.glob(os.path.join(A.out, '*')):
    os.remove(f)


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=1800).read()


rng = random.Random(time.time_ns())
words = open('/usr/share/dict/words').read().split() if os.path.exists('/usr/share/dict/words') else \
    [f'w{i}' for i in range(5000)]
text = f'[{rng.random()}] ' + ' '.join(rng.choice(words) for _ in range(A.words))
post('/start_profile', dict(output_dir=A.out, num_steps=1, activities=['GPU'], profile_prefix='prefill'))
t0 = time.time()
r = json.loads(post('/generate', dict(text=text, sampling_params=dict(temperature=0, max_new_tokens=1))))
print(json.dumps(dict(prompt_tokens=r['meta_info']['prompt_tokens'], ttft_s=round(time.time() - t0, 2))))
deadline = time.time() + 300
while not glob.glob(os.path.join(A.out, '*.trace.json*')) and time.time() < deadline:
    time.sleep(1)
time.sleep(3)
ev = json.load(gzip.open(sorted(glob.glob(os.path.join(A.out, '*.trace.json*')))[0], 'rt'))['traceEvents']
g = [e for e in ev if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
by = collections.Counter()
for e in g:
    by[e.get('args', {}).get('stream')] += e['dur']
main = by.most_common(1)[0][0]
m = sorted([e for e in g if e.get('args', {}).get('stream') == main], key=lambda e: e['ts'])
span = m[-1]['ts'] + m[-1]['dur'] - m[0]['ts']
gaps = [m[i + 1]['ts'] - (m[i]['ts'] + m[i]['dur']) for i in range(len(m) - 1)]
hist = collections.Counter()
for x in gaps:
    b = '<0.1ms' if x < 100 else '0.1-1ms' if x < 1000 else '1-10ms' if x < 10000 else '10-50ms' if x < 50000 else '>50ms'
    hist[b] += x
fam = collections.Counter()
for e in g:
    n = e['name']
    k = ('lkqmoe prefill' if any(s in n for s in ('_gate', '_down', '_unpack', '_reduce', '_copy_ids'))
         else 'copy' if e['cat'] != 'kernel' else 'attention/indexer' if any(s in n.lower() for s in ('mla', 'attn', 'indexer', 'mqa', 'compress'))
         else 'gemm' if any(s in n.lower() for s in ('gemm', 'cutlass', 'wmma')) else 'other')
    fam[k] += e['dur']
print(json.dumps(dict(span_ms=round(span / 1000, 1), main_busy_ms=round(by[main] / 1000, 1),
                      gaps_ms={k: round(v / 1000, 1) for k, v in hist.items()},
                      busy_by_family_ms={k: round(v / 1000, 1) for k, v in fam.most_common()})))
