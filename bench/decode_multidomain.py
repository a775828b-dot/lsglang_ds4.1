"""Decode speed on 20 fixed multi-domain prompts (code, math, writing, QA; Chinese and English), one
request at a time at temperature 0 and 0.7, 768 output tokens each. Reports per-request decode tok/s
and the DSpark accept length from the server's /metrics (verify calls per request).
Usage: decode_multidomain.py LABEL [--port 39504] [--model DeepSeek-V4.1-Flash-1017] [--out FILE]"""
import argparse, json, re, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument('label')
P.add_argument('--port', type=int, default=39504)
P.add_argument('--model', default='DeepSeek-V4.1-Flash-1017')
P.add_argument('--out')
A = P.parse_args()
PROMPTS = ['用 Python 实现一个线程安全的 LRU 缓存类，支持 get/put，并写单元测试。', '解释混合专家模型在推理时的访存特征，以及 CPU 卸载场景下的瓶颈。', '一个数列前三项是 2, 6, 14，满足 a(n) = 2*a(n-1) + c。求 c 和 a(10)，写出推导。', 'Write a Go HTTP middleware that adds request IDs, logs latency, and recovers from panics. Include tests.', "阅读这段报错并给出修复：TypeError: Cannot read properties of undefined (reading 'map') at TodoList.render", '写一篇 600 字左右的短文，介绍杭州西湖的四季景色。', 'Summarize the causes and consequences of the 2008 financial crisis in about 500 words.', '给出 Linux 下排查一台服务器 CPU 使用率 100% 的完整步骤和常用命令。', 'Translate the following into fluent English: 人工智能正在改变软件开发的方式，程序员需要学会与模型协作。', '设计一个电商订单系统的数据库表结构，说明索引和分库分表策略。', 'Explain the proof that there are infinitely many primes, then generalize to primes of the form 4k+3.', '用 Rust 写一个命令行工具，统计目录下各类文件的行数，并说明设计。', '写一首关于秋天的七言律诗，并逐句解释。', 'Compare PostgreSQL and MySQL for a write-heavy analytics workload; give concrete recommendations.', '分析《三体》中黑暗森林法则的逻辑前提和可能的漏洞。', 'Write a SQL query to find the top 3 customers by revenue per month, then explain window functions used.', '列出学习机器学习的 12 周计划，每周有目标、资料和练习。', 'Debug this Python: for i in range(len(xs)): if xs[i] == target: del xs[i]  — explain the bug and fix it.', '讲讲 TCP 三次握手与四次挥手，以及 TIME_WAIT 的作用。', 'Write a short story (about 500 words) about a lighthouse keeper who receives a letter from the future.']
URL = f'http://127.0.0.1:{A.port}'

def verify_calls():
    text = urllib.request.urlopen(URL + '/metrics', timeout=10).read().decode()
    m = re.search(r'^sglang:spec_verify_calls_total\{[^}]*\} ([0-9.e+]+)$', text, re.M)
    return float(m.group(1)) if m else 0.0


label = A.label
rounds = 1
rows = []
for r in range(rounds):
    for temp in (0.0, 0.7):
        for i, p in enumerate(PROMPTS):
            body = json.dumps(dict(model=A.model, messages=[{'role': 'user', 'content': p}],
                                   max_tokens=768, temperature=temp, stream=True,
                                   stream_options={'include_usage': True})).encode()
            req = urllib.request.Request(URL + '/v1/chat/completions', body, {'Content-Type': 'application/json'})
            v0 = verify_calls()
            t0 = time.time(); first = None; usage = None
            with urllib.request.urlopen(req, timeout=900) as resp:
                for raw in resp:
                    line = raw.decode().strip()
                    if not line.startswith('data:') or line == 'data: [DONE]':
                        continue
                    ev = json.loads(line[5:])
                    if first is None and ev.get('choices') and (ev['choices'][0].get('delta') or {}).get('content') is not None:
                        first = time.time()
                    if first is None and ev.get('choices') and (ev['choices'][0].get('delta') or {}).get('reasoning_content'):
                        first = time.time()
                    if ev.get('usage'):
                        usage = ev['usage']
            t1 = time.time()
            n = usage['completion_tokens']
            steps = verify_calls() - v0
            acc = n / steps if steps else 0
            dec = (n - 1) / (t1 - first) if first and n > 1 else 0
            rows.append(dict(temp=temp, i=i, tokens=n, ttft=round((first or t1) - t0, 3), decode_tps=round(dec, 2), steps=steps, accept=round(acc, 3)))
            print(f'{label} temp={temp} #{i} tokens={n} ttft={rows[-1]["ttft"]}s decode={dec:.1f} tok/s accept={acc:.2f}', flush=True)
summary = {}
for temp in (0.0, 0.7):
    rs = [x for x in rows if x['temp'] == temp]
    toks = sum(x['tokens'] - 1 for x in rs)
    secs = sum((x['tokens'] - 1) / x['decode_tps'] for x in rs if x['decode_tps'])
    steps = sum(x['steps'] for x in rs)
    summary[str(temp)] = dict(requests=len(rs), decode_tps=round(toks / secs, 2),
                              accept=round(sum(x['tokens'] for x in rs) / steps, 3) if steps else None)
print('SUMMARY', label, json.dumps(summary))
json.dump(dict(label=label, rows=rows, summary=summary),
          open(A.out or f'decode-{label}.json', 'w'), indent=1)
