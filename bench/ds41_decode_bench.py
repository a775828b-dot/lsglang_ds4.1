"""DeepSeek V4.1 decode speed over a fixed prompt set (greedy, ignore_eos), backend A/B.

Usage: python ds41_decode_bench.py [--port 39503] [--tokens 800]
Per prompt: steady-state tokens/s (stream timestamps after the first 3 s), speculative
accept length and the implied verify-step time; then the mean over prompts.
"""
import argparse, json, statistics as st, time, urllib.request, os

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=39503)
P.add_argument('--tokens', type=int, default=800)
A = P.parse_args()
PROMPTS = [
    '请用中文详细讲解现代操作系统的内存管理，包括分页、TLB、缺页中断、页面置换算法和NUMA，并给出代码示例。',
    'Implement a thread-safe LRU cache in Python with TTL support and full unit tests, then explain the design.',
    '写一篇关于长江流域历史变迁的长文，分朝代叙述，每段不少于200字。',
    'Solve step by step: a tank is filled by pipe A in 6 h and by pipe B in 9 h, and drained by C in 12 h. '
    'All three open at 8:00; when is it full? Then generalise to n pipes and prove your formula.',
    'Write a Rust implementation of a lock-free single-producer single-consumer ring buffer with documentation '
    'comments, benchmarks and a discussion of memory ordering.',
    '把下面的需求整理成详细的产品需求文档（PRD）：一个支持多人协作的在线白板，包含权限、版本历史、离线同步、导出。',
]


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ.get('DS41_TOKENIZER', 'deepseek-ai/DeepSeek-V4.1-Flash'), trust_remote_code=True)


def run(prompt):
    try:
        text = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    body = dict(text=text, stream=True, sampling_params=dict(temperature=0, max_new_tokens=A.tokens, ignore_eos=True))
    req = urllib.request.Request(f'http://127.0.0.1:{A.port}/generate', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    stamps, info = [], {}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            if raw.startswith(b'data:') and raw[5:].strip() != b'[DONE]':
                v = json.loads(raw[5:])
                stamps.append((time.time(), v['meta_info']['completion_tokens']))
                info = v['meta_info']
    return t0, stamps, info


rows = []
for prompt in PROMPTS:
    t0, stamps, info = run(prompt)
    steady = [(t, c) for t, c in stamps if t >= stamps[0][0] + 3]
    rate = (steady[-1][1] - steady[0][1]) / (steady[-1][0] - steady[0][0])
    acc = info.get('spec_accept_length') or 1.0
    rows.append(dict(prompt=prompt[:12], tokens_per_s=round(rate, 1), accept_length=round(acc, 2),
                     verify_step_ms=round(1000 * acc / rate, 1), ttft_s=round(stamps[0][0] - t0, 2)))
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
print(json.dumps(dict(mean_tokens_per_s=round(st.mean(r['tokens_per_s'] for r in rows), 2),
                      mean_accept=round(st.mean(r['accept_length'] for r in rows), 2),
                      mean_verify_step_ms=round(st.mean(r['verify_step_ms'] for r in rows), 1))), flush=True)
