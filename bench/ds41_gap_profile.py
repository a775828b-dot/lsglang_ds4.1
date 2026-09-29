"""What does the host do while the GPU idles between DeepSeek V4.1 verify steps?

Usage: python ds41_gap_profile.py [--port 39503] [--steps 6] [--out DIR]
Profiles a few decode iterations with CPU+GPU activity and Python stacks, finds the long
(>= 3 ms) idle gaps of the busiest GPU stream, and lists which CPU-side events (Python
functions, aten ops, CUDA runtime calls) overlap those gaps, by total overlapped time.
"""
import argparse, collections, glob, gzip, json, os, threading, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=39503)
P.add_argument('--steps', type=int, default=6)
P.add_argument('--out', default='/tmp/ds41-gap-profile')
P.add_argument('--trace', help='analyze an existing trace instead of profiling')
A = P.parse_args()
BASE = f'http://127.0.0.1:{A.port}'


def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


path = A.trace
if not path:
    os.makedirs(A.out, exist_ok=True)
    for f in glob.glob(os.path.join(A.out, '*')):
        os.remove(f)
    threading.Thread(target=lambda: post('/generate', dict(text='请详细介绍量子计算的发展历史和主要算法。',
                     sampling_params=dict(temperature=0, max_new_tokens=1200, ignore_eos=True))), daemon=True).start()
    time.sleep(6)
    post('/start_profile', dict(output_dir=A.out, num_steps=A.steps, activities=['CPU', 'GPU'], with_stack=True,
                                profile_prefix='gap'))
    deadline = time.time() + 600
    while not glob.glob(os.path.join(A.out, '*.trace.json*')) and time.time() < deadline:
        time.sleep(2)
    time.sleep(10)
    path = sorted(glob.glob(os.path.join(A.out, '*.trace.json*')))[0]
events = json.load(gzip.open(path, 'rt') if path.endswith('.gz') else open(path))['traceEvents']
gpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
by = collections.Counter()
for e in gpu:
    by[e.get('args', {}).get('stream')] += e['dur']
main = by.most_common(1)[0][0]
m = sorted([e for e in gpu if e.get('args', {}).get('stream') == main], key=lambda e: e['ts'])
gaps = [(m[i]['ts'] + m[i]['dur'], m[i + 1]['ts']) for i in range(len(m) - 1)
        if m[i + 1]['ts'] - (m[i]['ts'] + m[i]['dur']) >= 3000]
print(json.dumps(dict(trace=path, main_stream=main, long_gaps=len(gaps),
                      gap_ms=[round((b - a) / 1000, 2) for a, b in gaps])))
cpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('python_function', 'cpu_op', 'cuda_runtime',
                                                                    'cuda_driver', 'user_annotation')]
overlap = collections.Counter(); hits = collections.Counter()
for e in cpu:
    a, b = e['ts'], e['ts'] + e.get('dur', 0)
    for ga, gb in gaps:
        o = min(b, gb) - max(a, ga)
        if o > 0:
            key = (e['cat'], e['name'][:110], e.get('tid'))
            overlap[key] += o; hits[key] += 1
total = sum(b - a for a, b in gaps) or 1
print('share of gap time covered by each CPU event (inclusive; nested events overlap):')
for (cat, name, tid), v in overlap.most_common(60):
    print(f'{100 * v / total:6.1f}%  {v / len(gaps) / 1000:7.2f} ms/gap  x{hits[(cat, name, tid)] / len(gaps):5.1f}  tid={tid}  {cat}: {name}')
