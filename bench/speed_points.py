"""Prefill / decode speed at several input lengths against an OpenAI-compatible sglang server.

Usage:
  python speed_points.py --port 39503 --model DeepSeek-V4.1-Flash --tokenizer /path/to/model \
      --points 4096,16384,32768,65536,131072,262144,393216,500000 --output 1024 --out result.json
Each point: a filler document of about that many tokens with one fact buried at 10% depth
and a question about it at the end (a new random document per point, so no prefix-cache
reuse). Streams the answer with temperature 0 and ignore_eos, and reports prompt tokens,
TTFT, prefill tokens/s, decode tokens/s, speculative accept length (if the server reports
it), whether the fact was recalled, and peak VRAM.
"""
import argparse, json, random, subprocess, threading, time, urllib.request
from transformers import AutoTokenizer

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, required=True)
P.add_argument('--model', required=True, help='served model name')
P.add_argument('--tokenizer', required=True, help='model directory with the tokenizer')
P.add_argument('--points', required=True, help='comma-separated input lengths in tokens')
P.add_argument('--output', type=int, default=1024)
P.add_argument('--out')
P.add_argument('--seed', type=int, default=0, help='added to the document seed of every point (fresh text, no prefix-cache reuse)')
P.add_argument('--exact', action='store_true', help='rescale the document until its token count is within 0.05%% of the point')
P.add_argument('--stop-on-fail', action='store_true', help='stop at the first point that errors or does not recall the fact')
A = P.parse_args()
tok = AutoTokenizer.from_pretrained(A.tokenizer, trust_remote_code=True)
BASE = f'http://127.0.0.1:{A.port}'
TOPICS = ['memory allocator', 'scheduler', 'network stack', 'file system', 'compiler pass', 'GPU driver',
          'database index', 'cache policy', 'build system', 'test harness']


def document(tokens, seed):
    def build(n=None):
        rng = random.Random(seed)
        def paragraph(i):
            t = rng.choice(TOPICS)
            return (f'Log entry {i}: the {t} team reviewed ticket {rng.randint(1000, 99999)}. They measured '
                    f'{rng.randint(1, 999)} ms latency, changed {rng.randint(1, 60)} files and noted that the '
                    f'{rng.choice(TOPICS)} depends on the {rng.choice(TOPICS)}. Follow-up owner: engineer {rng.randint(1, 500)}.\n')
        code = f'{rng.randint(1000, 9999)}-{rng.choice(["ALPHA", "BRAVO", "DELTA", "OSCAR"])}'
        per = len(tok.encode(paragraph(0), add_special_tokens=False))
        n = n or max(1, int((tokens - 200) / per))
        parts = [paragraph(i) for i in range(n)]
        parts.insert(n // 10, f'IMPORTANT: the release code for project Lanternfish is {code}. Keep it for the final answer.\n')
        question = ('\nTask: this was a long agent session log. First state the release code for project '
                    'Lanternfish exactly, then write a detailed compaction summary of the session.')
        return ''.join(parts) + question, code, n
    text, code, n = build()
    for _ in range(4 if A.exact else 0):
        # the one-paragraph estimate is off by a few percent; rescale the paragraph count to the measured length
        have = len(tok.encode(text, add_special_tokens=False))
        if abs(have - tokens) <= max(64, tokens // 2000):
            break
        text, code, n = build(max(1, round(n * tokens / have)))
    return text, code


def vram():
    return int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                       text=True).split()[0])


def accept_length():
    try:
        with urllib.request.urlopen(BASE + '/get_server_info', timeout=30) as r:
            return (json.loads(r.read()).get('internal_states') or [{}])[0].get('avg_spec_accept_length')
    except Exception:
        return None


def run(tokens):
    text, code = document(tokens, seed=tokens + A.seed)
    body = dict(model=A.model, messages=[{'role': 'user', 'content': text}], stream=True, max_tokens=A.output,
                temperature=0, ignore_eos=True, stream_options={'include_usage': True})
    peak, stop = [vram()], threading.Event()
    def poll():
        while not stop.is_set():
            peak[0] = max(peak[0], vram()); time.sleep(0.2)
    threading.Thread(target=poll, daemon=True).start()
    req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(body).encode(),
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
    prompt, completion = usage.get('prompt_tokens') or 0, usage.get('completion_tokens') or 0
    acc = accept_length()
    return dict(target=tokens, prompt_tokens=prompt, completion_tokens=completion,
                ttft_s=first and round(first - t0, 2),
                prefill_tok_s=first and prompt and round(prompt / (first - t0)),
                decode_tok_s=first and completion and round(completion / max(t1 - first, 1e-6), 1),
                accept_length=acc and round(acc, 2), recall=code in ''.join(out), vram_peak_mib=peak[0], error=error)


rows = []
for n in [int(x) for x in A.points.split(',')]:
    rows.append(run(n))
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    if A.stop_on_fail and (rows[-1]['error'] or not rows[-1]['recall']):
        break
if A.out:
    json.dump(rows, open(A.out, 'w'), indent=1)
