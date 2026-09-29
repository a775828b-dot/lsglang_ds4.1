"""Speculative accept length over a larger prompt set (backend A/B on DeepSeek V4.1).

Usage: python ds41_accept_bench.py [--port 39503] [--tokens 512] [--out FILE]
24 prompts (Chinese/English prose, code, math, structured output, agent-style), greedy,
ignore_eos. Per prompt: completion tokens, verify steps, accept length, tokens/s.
"""
import argparse, json, statistics as st, time, urllib.request, os

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=39503)
P.add_argument('--tokens', type=int, default=512)
P.add_argument('--out')
A = P.parse_args()
PROMPTS = [
    '请解释TCP三次握手和四次挥手的过程，并说明TIME_WAIT存在的原因。',
    '用Python实现一个支持插入、删除和查找第k小元素的平衡二叉搜索树，并附测试。',
    'Explain how transformers use attention, then derive the complexity of self-attention.',
    'Write a Go HTTP server with graceful shutdown, structured logging and a health endpoint.',
    '写一份新员工入职指南，包含第一周的日程安排、需要开通的账号和常见问题。',
    'Prove that the square root of 2 is irrational, then generalise to square roots of non-square integers.',
    '比较Redis和Memcached的架构差异、持久化方式和适用场景，用表格总结。',
    'Write a SQL schema for an online bookstore and five analytical queries with explanations.',
    '请把下面的会议纪要整理成行动项列表：讨论了Q3目标、预算调整、招聘计划和上线时间表。',
    'Implement Dijkstra and A* in C++, compare them on a grid, and discuss heuristics.',
    '解释一下量子纠缠，并用通俗的比喻讲给高中生听。',
    'Draft a JSON schema for a travel itinerary and give three valid example documents.',
    '写一个Shell脚本，定期备份指定目录到远程服务器，保留最近7份并记录日志。',
    'Summarise the causes and consequences of the 2008 financial crisis in detail.',
    '请设计一个秒杀系统的整体架构，包括限流、缓存、队列和数据库扣减库存的方案。',
    'Write a React component for a sortable, filterable data table with TypeScript types.',
    '计算：一个等比数列前n项和为S_n=3^n-1，求通项公式并证明。',
    'You are an agent. Plan the steps to migrate a monolith to microservices, listing tools you would call.',
    '写一首关于秋天的现代诗，然后逐句解释其意象。',
    'Explain Rust ownership, borrowing and lifetimes with runnable examples.',
    '分析一下电动汽车电池回收产业链的现状和挑战。',
    'Write unit tests in pytest for a function that parses ISO-8601 durations.',
    '请用Markdown写一份API文档，描述用户注册、登录、刷新令牌三个接口。',
    'Describe the CAP theorem and how Cassandra, MongoDB and Spanner make trade-offs.',
]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ.get('DS41_TOKENIZER', 'deepseek-ai/DeepSeek-V4.1-Flash'), trust_remote_code=True)
rows = []
for p in PROMPTS:
    try:
        text = tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
    except Exception:
        text = p
    body = dict(text=text, sampling_params=dict(temperature=0, max_new_tokens=A.tokens, ignore_eos=True))
    req = urllib.request.Request(f'http://127.0.0.1:{A.port}/generate', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    dt = time.time() - t0
    meta = r['meta_info']
    n = meta['completion_tokens']; acc = meta.get('spec_accept_length')
    rows.append(dict(prompt=p[:14], tokens=n, seconds=round(dt, 2), tok_s=round(n / dt, 1), accept=acc and round(acc, 3)))
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
summary = dict(prompts=len(rows), mean_accept=round(st.mean(r['accept'] for r in rows if r['accept']), 3),
               median_accept=round(st.median(r['accept'] for r in rows if r['accept']), 3),
               mean_tok_s=round(st.mean(r['tok_s'] for r in rows), 2), total_tok_s=round(
                   sum(r['tokens'] for r in rows) / sum(r['seconds'] for r in rows), 2))
print(json.dumps(summary), flush=True)
if A.out:
    open(A.out, 'w').write(json.dumps(dict(rows=rows, summary=summary), ensure_ascii=False, indent=1))
