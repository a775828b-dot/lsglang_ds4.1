"""Long-prompt TTFT and GPU memory peak on a running DeepSeek V4.1 server.

Usage: python ds41_prefill_bench.py [--port 39503] [--tokens 32768 8192] [--repeats 2]
Each request uses a fresh random prefix (no radix-cache hit) and max_new_tokens=1.
nvidia-smi is sampled every 100 ms for the peak used memory during the request.
"""
import argparse, json, random, subprocess, threading, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=39503)
P.add_argument('--tokens', type=int, nargs='+', default=[32768, 8192])
P.add_argument('--repeats', type=int, default=2)
A = P.parse_args()
words = open('/usr/share/dict/words').read().split() if __import__('os').path.exists('/usr/share/dict/words') else \
    [f'w{i}' for i in range(5000)]


def used_mib():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    return int(out.split()[0])


def request(text):
    body = json.dumps(dict(text=text, sampling_params=dict(max_new_tokens=1, temperature=0))).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{A.port}/generate', body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())


for n in A.tokens:
    for rep in range(A.repeats):
        rng = random.Random(time.time_ns())
        text = f'[{rng.random()}] ' + ' '.join(rng.choice(words) for _ in range(int(n * 0.55)))
        peak = [used_mib()]; stop = threading.Event()
        def poll():
            while not stop.is_set():
                peak[0] = max(peak[0], used_mib()); time.sleep(0.1)
        th = threading.Thread(target=poll); th.start()
        t0 = time.perf_counter(); r = request(text); dt = time.perf_counter() - t0
        stop.set(); th.join()
        meta = r.get('meta_info', {})
        print(json.dumps(dict(target_tokens=n, prompt_tokens=meta.get('prompt_tokens'), ttft_s=round(dt, 3),
                              tok_per_s=round(meta.get('prompt_tokens', 0) / dt, 1), vram_peak_mib=peak[0])), flush=True)
