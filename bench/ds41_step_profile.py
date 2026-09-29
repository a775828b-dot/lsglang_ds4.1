"""Where does a DeepSeek V4.1 decode verify step go? Live GPU trace + lkqmoe CPU timing.

Usage: python ds41_step_profile.py [--port 39503] [--steps 30] [--out DIR]
While one greedy request decodes, profiles `--steps` scheduler iterations (GPU activity),
then reports per step: GPU busy (interval union), GPU idle split into short (<50 us,
launch/sync) and long gaps (waiting on CPU experts), busy time by kernel family, the top
kernels, and the CPU expert time per step from the lkqmoe stats delta (if present).
"""
import argparse, collections, glob, gzip, json, os, re, threading, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=39503)
P.add_argument('--steps', type=int, default=30)
P.add_argument('--out', default='/tmp/ds41-step-profile')
P.add_argument('--stats', default=os.environ.get('LKQMOE_STATS_DIR', 'run/lkqmoe-stats'))
A = P.parse_args()
BASE = f'http://127.0.0.1:{A.port}'
os.makedirs(A.out, exist_ok=True)
for f in glob.glob(os.path.join(A.out, '*')):
    os.remove(f)


def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def cpu_stats():
    rows = collections.Counter()
    best = None
    for f in sorted(glob.glob(os.path.join(A.stats, 'stats-*.json')), key=os.path.getmtime):
        try:
            data = json.load(open(f))
        except (OSError, ValueError):
            continue
        if data.get('instances'):
            best = data
    if best is None:
        return rows
    for inst in best.get('instances', []):
        for m, t in (inst.get('decode_timing') or {}).items():
            rows[(m, 'calls')] += t['calls']; rows[(m, 'us')] += t['calls'] * t['wall_us']
            rows[(m, 'gate')] += t['calls'] * t['gate_us']; rows[(m, 'down')] += t['calls'] * t['down_us']
    return rows


def spec_info():
    with urllib.request.urlopen(BASE + '/get_server_info', timeout=30) as r:
        return (json.loads(r.read()).get('internal_states') or [{}])[0]


prompt = '请用中文详细讲解现代操作系统的内存管理，包括分页、TLB、缺页中断、页面置换算法和NUMA，并给出代码示例。'
done = threading.Event()
def generate():
    post('/generate', dict(text=prompt, sampling_params=dict(temperature=0, max_new_tokens=1500, ignore_eos=True)))
    done.set()
threading.Thread(target=generate, daemon=True).start()
time.sleep(6)
before = cpu_stats()
post('/start_profile', dict(output_dir=A.out, num_steps=A.steps, activities=['GPU'], profile_prefix='ds41'))
deadline = time.time() + 300
while not glob.glob(os.path.join(A.out, '*.trace.json*')) and time.time() < deadline:
    time.sleep(1)
time.sleep(5)
after = cpu_stats()
done.wait(600)

path = sorted(glob.glob(os.path.join(A.out, '*.trace.json*')))[0]
events = json.load(gzip.open(path, 'rt') if path.endswith('.gz') else open(path))['traceEvents']
gpu = sorted([e for e in events if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')],
             key=lambda e: e['ts'])
merged = []
for e in gpu:
    a, b = e['ts'], e['ts'] + e['dur']
    if merged and a <= merged[-1][1]:
        merged[-1][1] = max(b, merged[-1][1])
    else:
        merged.append([a, b])
span = merged[-1][1] - merged[0][0]
busy = sum(b - a for a, b in merged)
gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
short = sum(g for g in gaps if g < 50); long_ = [g for g in gaps if g >= 50]


def family(name, cat):
    n = name.lower()
    if cat != 'kernel': return 'copy/memset'
    if 'lkqmoe' in n or '_pack' in n or '_unpack' in n: return 'lkqmoe bridge'
    if any(k in n for k in ('moe', 'expert', 'grouped', 'fp4_gemm', 'nvfp4', 'topk', 'router')): return 'MoE (GPU layers, router, top-k)'
    if any(k in n for k in ('attn', 'attention', 'mla', 'flash', 'fmha', 'indexer', 'mqa', 'paged', 'sparse', 'compress')):
        return 'attention / indexer / compressor'
    if any(k in n for k in ('gemm', 'gemv', 'cutlass', 'cublas', 'xmma', 'sm90', 'sm100', 'sm120', 'matmul', 'deep_gemm', 'wgmma')):
        return 'dense GEMM/GEMV'
    if any(k in n for k in ('norm', 'rms')): return 'norm'
    if any(k in n for k in ('sample', 'softmax', 'argmax', 'verify', 'accept', 'tree', 'spec')): return 'sampling / verify'
    if any(k in n for k in ('hc_', 'mhc', 'hyper')): return 'hyper-connection mix'
    return 'other elementwise/misc'


fam, top = collections.Counter(), collections.Counter()
count = collections.Counter()
for e in gpu:
    f = family(e['name'], e['cat']); fam[f] += e['dur']; top[e['name'][:90]] += e['dur']; count[e['name'][:90]] += 1
steps = A.steps
per = lambda us: round(us / steps / 1000, 2)
cpu = {}
for m in ('6', '8', '4', '2', '1'):
    calls = after[(m, 'calls')] - before[(m, 'calls')]
    if calls > 0:
        cpu[m] = dict(calls=calls, layer_us=round((after[(m, 'us')] - before[(m, 'us')]) / calls, 1),
                      gate_us=round((after[(m, 'gate')] - before[(m, 'gate')]) / calls, 1),
                      down_us=round((after[(m, 'down')] - before[(m, 'down')]) / calls, 1))
by_stream = collections.Counter()
for e in gpu:
    by_stream[e.get('args', {}).get('stream')] += e['dur']
main = by_stream.most_common(1)[0][0]
ms_ = sorted([e for e in gpu if e.get('args', {}).get('stream') == main], key=lambda e: e['ts'])
mgaps = [ms_[i + 1]['ts'] - (ms_[i]['ts'] + ms_[i]['dur']) for i in range(len(ms_) - 1)]
cpu_wait = sum(x for x in mgaps if 900 <= x < 2500); inter_step = sum(x for x in mgaps if x >= 2500)
report = dict(trace=path, steps=steps, step_ms=per(span), gpu_busy_ms=per(busy),
              main_cpu_wait_ms=per(cpu_wait), main_inter_step_ms=per(inter_step),
              gpu_idle_long_gaps_ms=per(sum(long_)), long_gaps_per_step=round(len(long_) / steps, 1),
              gpu_idle_short_gaps_ms=per(short),
              busy_by_family_ms={k: per(v) for k, v in fam.most_common()},
              top_kernels_ms=[(k, per(v), round(count[k] / steps, 1)) for k, v in top.most_common(20)],
              cpu_expert_calls_in_window=cpu, spec=spec_info().get('avg_spec_accept_length'))
print(json.dumps(report, ensure_ascii=False, indent=1))
