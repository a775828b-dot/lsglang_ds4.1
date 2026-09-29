"""DeepSeek V4.1 routine speed check: three input lengths against the official baseline.

Usage: python ds41_3point.py [--port 39503] [--output 1024] [--tag NAME]
Inputs of ~8.2K / 78.7K / 219.7K tokens (fresh random text each run, no prefix-cache hit),
1024 output tokens. Prefill = input tokens / TTFT; decode = output tokens / decode time.
Baseline: official DeepSeek V4.1 + lk_moe (2026-09-17): prefill 1.3K / 2.47K / 2.3K tok/s;
decode ~45 falling to ~41 tok/s. The ~500K capacity run is for final acceptance only
(ds41_longctx_test.py).
"""
import argparse, json, os, subprocess, sys

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=39503)
P.add_argument('--output', type=int, default=1024)
P.add_argument('--tag', default='')
A = P.parse_args()
POINTS = [int(v) for v in os.environ.get('POINTS', '8200,78700,219700').split(',')]
BASE_ALL = {8200: (1300, 45), 78700: (2470, 43), 219700: (2300, 42)}
BASE = {n: BASE_ALL[n] for n in POINTS}
here = os.path.dirname(os.path.abspath(__file__))
rows = []
for n, (base_prefill, base_decode) in BASE.items():
    out = subprocess.run([sys.executable, os.path.join(here, 'ds41_longctx_test.py'), '--port', str(A.port),
                          '--input', str(n), '--output', str(A.output)], capture_output=True, text=True).stdout
    d = json.loads([l for l in out.splitlines() if l.startswith('{')][-1])
    prefill = d['prompt_tokens'] / d['ttft_s']
    rows.append(dict(target=n, input=d['prompt_tokens'], ttft_s=d['ttft_s'], prefill_tok_s=round(prefill),
                     prefill_vs_official=f'{100 * (prefill / base_prefill - 1):+.0f}%', decode_tok_s=d['decode_tok_s'],
                     decode_vs_official=f'{100 * (d["decode_tok_s"] / base_decode - 1):+.0f}%',
                     accept=d.get('accept_length'), verify_step_ms=d.get('verify_step_ms'),
                     recall=d['needle_found'], vram_peak_mib=d['vram_peak_mib'], alive=d['server_alive']))
    print(json.dumps(rows[-1]), flush=True)
    if not d['server_alive']:
        break
if A.tag:
    path = os.path.join(here, '..', 'runs', f'ds41-3point-{A.tag}.json')
    json.dump(rows, open(path, 'w'), indent=1)
