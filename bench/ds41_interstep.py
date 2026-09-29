"""Break down the gap between DeepSeek V4.1 verify steps from a ds41_step_profile.py trace.

Usage: python ds41_interstep.py [TRACE]   (default: newest /tmp/ds41-step-profile/*.trace.json.gz)
For every main-stream gap >= 2.5 ms: GPU time on other streams inside it (the draft), which
kernels ran there, and the time with no GPU activity at all (host-side scheduling).
"""
import collections, glob, gzip, json, os, sys

path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob('/tmp/ds41-step-profile/*.trace.json*'), key=os.path.getmtime)
events = json.load(gzip.open(path, 'rt') if path.endswith('.gz') else open(path))['traceEvents']
gpu = sorted([e for e in events if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')],
             key=lambda e: e['ts'])
by_stream = collections.Counter()
for e in gpu:
    by_stream[e.get('args', {}).get('stream')] += e['dur']
main = by_stream.most_common(1)[0][0]
ms = [e for e in gpu if e.get('args', {}).get('stream') == main]
gaps = [(ms[i]['ts'] + ms[i]['dur'], ms[i + 1]['ts'], ms[i]['name'][:60], ms[i + 1]['name'][:60])
        for i in range(len(ms) - 1) if ms[i + 1]['ts'] - (ms[i]['ts'] + ms[i]['dur']) >= 2500
        and ms[i]['name'] != '_pack']  # _pack -> _unpack is a long CPU-expert wait, not a step boundary
total = busy_other = 0
per_gap = []
kern = collections.Counter(); streams = collections.Counter(); kcount = collections.Counter()
edges = collections.Counter()
for a, b, before, after in gaps:
    total += b - a
    edges[(before, after)] += 1
    iv = []
    for e in gpu:
        s, t = e['ts'], e['ts'] + e['dur']
        if t <= a or s >= b:
            continue
        s, t = max(s, a), min(t, b)
        iv.append((s, t)); kern[e['name'][:80]] += t - s; kcount[e['name'][:80]] += 1
        streams[e.get('args', {}).get('stream')] += t - s
    iv.sort(); merged = []
    for s, t in iv:
        if merged and s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], t)
        else: merged.append([s, t])
    b_ = sum(t - s for s, t in merged); busy_other += b_; per_gap.append((b - a, b_))
n = len(gaps)
print(json.dumps(dict(trace=path, main_stream=main, gaps=n, gap_ms=round(total / n / 1000, 2),
                      gpu_busy_in_gap_ms=round(busy_other / n / 1000, 2),
                      gpu_idle_in_gap_ms=round((total - busy_other) / n / 1000, 2),
                      median_gap_ms=round(sorted(g for g, _ in per_gap)[n // 2] / 1000, 2),
                      median_idle_ms=round(sorted(g - x for g, x in per_gap)[n // 2] / 1000, 2),
                      median_busy_ms=round(sorted(x for _, x in per_gap)[n // 2] / 1000, 2),
                      streams_in_gap_ms={str(k): round(v / n / 1000, 2) for k, v in streams.most_common()}), indent=1))
print('gap edges (last main kernel before -> first after):')
for (x, y), c in edges.most_common(4):
    print(f'  x{c}: {x}  ->  {y}')
print('kernels inside the gaps (ms per gap, launches per gap):')
for k, v in kern.most_common(25):
    print(f'  {v / n / 1000:6.3f}  x{kcount[k] / n:5.1f}  {k}')
