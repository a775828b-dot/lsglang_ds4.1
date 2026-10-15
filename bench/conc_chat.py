#!/usr/bin/env python3
"""Concurrent decode speed through /v1/chat/completions (servers without an HF chat template, e.g. DeepSeek V4.1).
Same method as conc_bench.py: C streaming requests started together, per-request decode = (tokens - 1) /
(last - first content chunk), aggregate = all tokens / (last chunk - first chunk of the round).
Usage: conc_chat.py --port 39503 --model DeepSeek-V4.1-Flash [--levels 1,2] [--rounds 4] [--max-new 512] [--out F]"""
import argparse
import http.client
import json
import statistics as st
import threading
import time

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=39503)
P.add_argument('--model', default='DeepSeek-V4.1-Flash')
P.add_argument('--levels', default='1,2')
P.add_argument('--rounds', type=int, default=4)
P.add_argument('--max-new', type=int, default=512)
P.add_argument('--temperature', type=float, default=1.0)
P.add_argument('--out')
A = P.parse_args()
PROMPTS = [
    '请解释为什么天空是蓝色的，以及日落时天空为什么会变红。分点说明，语言通俗。',
    '用 Python 写一个异步网页下载器，要求有并发限制、失败重试和断点续传，并用中文解释关键设计。',
    'Write a short technical blog post comparing Rust and Go for building network services, with code examples.',
    '一个水池有进水管和出水管，进水管单独 6 小时注满，出水管单独 9 小时放完。两管同时开，多少小时注满？给出详细推理，再推广到一般情况。',
    'Implement an LRU cache in TypeScript with O(1) get/put, generics, and unit tests using vitest. Explain the design briefly.',
    '设计一个支持多租户的 API 网关限流方案：说明令牌桶与滑动窗口的取舍，给出 Redis Lua 脚本实现，并讨论分布式一致性问题。',
    'Explain the CAP theorem with concrete examples of databases in each category, then discuss PACELC.',
    '帮我写一封给客户的道歉邮件：我们的服务昨晚中断了 3 小时，说明原因（数据库主从切换失败）、补偿方案和改进措施。',
]


def one(text, res, i):
    body = json.dumps({'model': A.model, 'messages': [{'role': 'user', 'content': text}], 'stream': True,
                       'stream_options': {'include_usage': True}, 'max_tokens': A.max_new, 'ignore_eos': True,
                       'temperature': A.temperature, 'top_p': 0.95}).encode()
    conn = http.client.HTTPConnection('127.0.0.1', A.port, timeout=3600)
    t0 = time.time()
    conn.request('POST', '/v1/chat/completions', body=body, headers={'Content-Type': 'application/json'})
    resp = conn.getresponse()
    first = last = None
    usage, buf = {}, b''
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b'\n\n' in buf:
            event, buf = buf.split(b'\n\n', 1)
            if not event.startswith(b'data: ') or event == b'data: [DONE]':
                continue
            d = json.loads(event[6:])
            if d.get('usage'):
                usage = d['usage']
            ch = (d.get('choices') or [{}])[0].get('delta') or {}
            if ch.get('content') or ch.get('reasoning_content'):
                now = time.time()
                first = first or now
                last = now
    n = usage.get('completion_tokens', 0)
    res[i] = dict(tokens=n, ttft=(first or t0) - t0, first=first, last=last,
                  decode=(n - 1) / (last - first) if first and last and last > first else 0.0)


summary, k = {}, 0
for c in [int(x) for x in A.levels.split(',')]:
    rows = []
    for r in range(A.rounds):
        res = [None] * c
        th = [threading.Thread(target=one, args=(PROMPTS[(k + i) % len(PROMPTS)], res, i)) for i in range(c)]
        k += c
        for t in th:
            t.start()
        for t in th:
            t.join()
        agg = sum(x['tokens'] for x in res) / (max(x['last'] for x in res) - min(x['first'] for x in res))
        rows.append(dict(round=r, aggregate=agg, requests=res))
        print(f'C={c} round {r}: per-request decode ' + ' / '.join(f"{x['decode']:.1f}" for x in res) +
              f' tok/s, aggregate {agg:.1f} tok/s, TTFT ' + ' / '.join(f"{x['ttft']:.2f}" for x in res) + ' s', flush=True)
    summary[c] = dict(per_request=round(st.mean(x['decode'] for row in rows for x in row['requests']), 1),
                      aggregate=round(st.mean(row['aggregate'] for row in rows), 1),
                      ttft=round(st.mean(x['ttft'] for row in rows for x in row['requests']), 2), rows=rows)
    print(f'== C={c}: per-request {summary[c]["per_request"]} tok/s, aggregate {summary[c]["aggregate"]} tok/s, '
          f'TTFT {summary[c]["ttft"]} s', flush=True)
if A.out:
    json.dump(dict(max_new=A.max_new, summary=summary), open(A.out, 'w'), ensure_ascii=False, indent=1)
