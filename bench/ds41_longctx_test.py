"""DeepSeek V4.1 long-context acceptance: ~500K input tokens + 2K output (agent compaction).

Usage: python ds41_longctx_test.py [--port 39503] [--input 500000] [--output 2000]
A fact is buried at 10% depth of a filler document and asked for at the end. Reports
prompt tokens, TTFT, decode tokens/s at that context, whether the answer is right,
VRAM peak (nvidia-smi, 200 ms), minimum host MemAvailable, and whether the server survived.
"""
import argparse, json, random, subprocess, threading, time, urllib.request, os
from transformers import AutoTokenizer

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=39503)
P.add_argument('--input', type=int, default=500000)
P.add_argument('--output', type=int, default=2000)
A = P.parse_args()
tok = AutoTokenizer.from_pretrained(os.environ.get('DS41_TOKENIZER', 'deepseek-ai/DeepSeek-V4.1-Flash'), trust_remote_code=True)
rng = random.Random(20260929)
topics = ['memory allocator', 'scheduler', 'network stack', 'file system', 'compiler pass', 'GPU driver',
          'database index', 'cache policy', 'build system', 'test harness']
def paragraph(i):
    t = rng.choice(topics)
    return (f'Log entry {i}: the {t} team reviewed ticket {rng.randint(1000, 99999)}. They measured '
            f'{rng.randint(1, 999)} ms latency, changed {rng.randint(1, 60)} files and noted that the '
            f'{rng.choice(topics)} depends on the {rng.choice(topics)}. Follow-up owner: engineer {rng.randint(1, 500)}.\n')
code = f'{rng.randint(1000, 9999)}-{rng.choice(["ALPHA", "BRAVO", "DELTA", "OSCAR"])}'
needle = f'IMPORTANT: the release code for project Lanternfish is {code}. Keep it for the final answer.\n'
question = ('\nTask: this was a long agent session log. First state the release code for project '
            'Lanternfish exactly, then write a detailed compaction summary of the session.')
per = len(tok.encode(paragraph(0), add_special_tokens=False))
n = int((A.input - 200) / per)
parts = [paragraph(i) for i in range(n)]
parts.insert(n // 10, needle)
text = ''.join(parts)
body = dict(model='DeepSeek-V4.1-Flash', messages=[{'role': 'user', 'content': text + question}], stream=True,
            max_tokens=A.output, temperature=0, ignore_eos=True, stream_options={'include_usage': True})


def vram():
    return int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).split()[0])
def avail():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) / 2**20
peak, low, stop = [vram()], [avail()], threading.Event()
def poll():
    while not stop.is_set():
        peak[0] = max(peak[0], vram()); low[0] = min(low[0], avail()); time.sleep(0.2)
threading.Thread(target=poll, daemon=True).start()

req = urllib.request.Request(f'http://127.0.0.1:{A.port}/v1/chat/completions', json.dumps(body).encode(),
                             {'Content-Type': 'application/json'})
t0 = time.time(); first = None; out = []; usage = {}; error = None
try:
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            if not raw.startswith(b'data:') or raw[5:].strip() == b'[DONE]':
                continue
            v = json.loads(raw[5:])
            usage = v.get('usage') or usage
            for c in v.get('choices', []):
                d = c.get('delta', {})
                piece = (d.get('content') or '') + (d.get('reasoning_content') or '')
                if piece:
                    first = first or time.time(); out.append(piece)
except Exception as exc:
    error = repr(exc)
t1 = time.time(); stop.set()
answer = ''.join(out)
try:
    urllib.request.urlopen(f'http://127.0.0.1:{A.port}/health', timeout=10); alive = True
except Exception:
    alive = False
completion = usage.get('completion_tokens') or 0
try:
    with urllib.request.urlopen(f'http://127.0.0.1:{A.port}/get_server_info', timeout=30) as r:
        accept = (json.loads(r.read()).get('internal_states') or [{}])[0].get('avg_spec_accept_length')
except Exception:
    accept = None
print(json.dumps(dict(prompt_tokens=usage.get('prompt_tokens'), completion_tokens=completion,
                      ttft_s=first and round(first - t0, 1),
                      decode_tok_s=first and completion and round(completion / max(t1 - first, 1e-6), 1),
                      accept_length=accept and round(accept, 2),
                      verify_step_ms=accept and first and completion and round(1000 * accept * max(t1 - first, 1e-6) / completion, 1),
                      needle_found=code in answer, answer_head=answer[:160], vram_peak_mib=peak[0],
                      min_mem_available_gib=round(low[0], 1), server_alive=alive, error=error),
                 ensure_ascii=False), flush=True)
