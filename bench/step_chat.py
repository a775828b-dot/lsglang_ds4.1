#!/usr/bin/env python3
"""Decode step time at batch size C, independent of the draft acceptance (which varies with the text).
C greedy (temperature 0) streaming chat requests start together; with speculative decoding every verify step
emits one stream chunk, so the median interval between a request's chunks while all C requests are still
generating is the step time of a batch of C. Also reports tokens per step (acceptance) and aggregate tok/s.
Greedy outputs repeat across server configs, so A/B runs route the same tokens.
Usage: step_chat.py --port 39503 [--levels 1,2] [--rounds 4] [--max-new 512] [--out F]"""
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
                       'temperature': 0.0}).encode()
    conn = http.client.HTTPConnection('127.0.0.1', A.port, timeout=3600)
    conn.request('POST', '/v1/chat/completions', body=body, headers={'Content-Type': 'application/json'})
    resp = conn.getresponse()
    times, usage, buf = [], {}, b''
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
                times.append(time.time())
    res[i] = dict(tokens=usage.get('completion_tokens', 0), times=times)


summary, k = {}, 0
for c in [int(x) for x in A.levels.split(',')]:
    steps, rows = [], []
    for r in range(A.rounds):
        res = [None] * c
        th = [threading.Thread(target=one, args=(PROMPTS[(k + i) % len(PROMPTS)], res, i)) for i in range(c)]
        k += c
        for t in th:
            t.start()
        for t in th:
            t.join()
        lo, hi = max(x['times'][0] for x in res), min(x['times'][-1] for x in res)
        mine = []
        for x in res:   # intervals while every request of the round is generating
            ts = [t for t in x['times'] if lo <= t <= hi]
            mine += [b - a for a, b in zip(ts, ts[1:])]
        steps += mine
        agg = sum(x['tokens'] for x in res) / (max(x['times'][-1] for x in res) - min(x['times'][0] for x in res))
        per_step = st.mean(x['tokens'] / max(len(x['times']), 1) for x in res)
        rows.append(dict(round=r, step_ms=st.median(mine) * 1e3, tokens_per_step=per_step, aggregate=agg))
        print(f'C={c} round {r}: step {rows[-1]["step_ms"]:.1f} ms (median of {len(mine)}), '
              f'{per_step:.2f} tokens/step, aggregate {agg:.1f} tok/s', flush=True)
    summary[c] = dict(step_ms=round(st.median(steps) * 1e3, 1), step_mean_ms=round(st.mean(steps) * 1e3, 1),
                      tokens_per_step=round(st.mean(r['tokens_per_step'] for r in rows), 2),
                      aggregate=round(st.mean(r['aggregate'] for r in rows), 1), rows=rows)
    print(f'== C={c}: step median {summary[c]["step_ms"]} ms, mean {summary[c]["step_mean_ms"]} ms, '
          f'{summary[c]["tokens_per_step"]} tokens/step, aggregate {summary[c]["aggregate"]} tok/s', flush=True)
if A.out:
    json.dump(dict(max_new=A.max_new, summary=summary), open(A.out, 'w'), ensure_ascii=False, indent=1)
