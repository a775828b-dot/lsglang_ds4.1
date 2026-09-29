"""Tune sglang's Triton W8A8 block-FP8 GEMM (block 32x32) for the DeepSeek V4.1 shapes on this GPU.

The stock tuner (benchmark/kernels/quantization/tuning_block_wise_kernel.py) has an empty search
space for block_k=32 and does not call the production function. This one times the production
`w8a8_block_fp8_matmul_triton` with a forced config (the module's config lookup is patched),
checks each winner against the default config's output, and writes the JSON the service reads.

Usage (service stopped, GPU free):
  PYTHONPATH=<source>/python python ds41_fp8_tune.py --out DIR [--shapes N:K ...] [--small-only]
Every file covers small M (decode, verify, draft) and large M (prefill), since the service picks
the entry with the nearest M for any batch.
"""
import argparse, itertools, json, math, os, sys, time
import torch

P = argparse.ArgumentParser()
P.add_argument('--out', required=True)
P.add_argument('--shapes', nargs='*', default=['1280:5120', '1536:5120', '1792:5120', '25600:6144', '32768:1280',
                                                '4096:1280', '4608:5120', '5120:15360', '5120:2304', '5120:8192',
                                                '512:5120'])
P.add_argument('--small', type=int, nargs='*', default=[1, 2, 4, 5, 6, 8, 16, 32])
P.add_argument('--large', type=int, nargs='*', default=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
P.add_argument('--small-only', action='store_true')
P.add_argument('--merge', action='store_true', help='start from the file already in --out; only the listed M are retuned')
P.add_argument('--shortlist-from', type=int, default=2048,
               help='from this M on, time only the 16 fastest configs of the previous M (large M is compute-bound)')
P.add_argument('--skip-done', action='store_true', help='skip shapes whose report for this M list already exists')
A = P.parse_args()

from sglang.kernels.ops.quantization import fp8_kernel as fk
BLOCK = [32, 32]
dev = torch.device('cuda')
name = torch.cuda.get_device_name().replace(' ', '_')

forced = {}
fk.get_w8a8_block_fp8_configs = lambda N, K, bn, bk: forced.get('cfg')  # module-global lookup, patched


def space(m):
    out = []
    if m <= 32:
        grid = itertools.product([16, 32], [16, 32, 64, 128], [32, 64, 128, 256], [8], [2, 4, 8], [2, 3, 4, 5, 6])  # one tile row: GROUP_SIZE_M is moot
    else:
        grid = itertools.product([32, 64, 128, 256] if m >= 256 else [32, 64, 128], [64, 128, 256], [32, 64, 128],
                                 [8, 32], [4, 8], [2, 3, 4])
    for bm, bn, bk, g, w, s in grid:
        if bm > max(16, 2 ** math.ceil(math.log2(m))):  # no tiles taller than the batch
            continue
        out.append(dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=g, num_warps=w, num_stages=s))
    return out


def timed(fn, iters):
    # fn(i) uses weight copy i % copies: the copies together exceed L2, so every call reads its
    # weights from DRAM as in the model (one weight read per layer, evicted in between)
    fn(0); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            for i in range(iters): fn(i)
    except Exception:
        g = None
    best = float('inf')
    for _ in range(3):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        if g is not None: g.replay()
        else:
            for i in range(iters): fn(i)
        e.record(); e.synchronize()
        best = min(best, s.elapsed_time(e) * 1000 / iters)
    return best


def run(Aq, B, As, Bs, cfg):
    forced['cfg'] = None if cfg is None else {Aq.shape[0]: cfg}
    return fk.w8a8_block_fp8_matmul_triton(Aq, B, As, Bs, BLOCK, output_dtype=torch.bfloat16)


os.makedirs(A.out, exist_ok=True)
for shape in A.shapes:
    N, K = map(int, shape.split(':'))
    torch.manual_seed(N * 7 + K)
    copies = max(1, math.ceil(320e6 / (N * K)))
    Bc = [(torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fn) for _ in range(copies)]
    Bsc = [torch.rand(N // 32, K // 32, device=dev) * 1e-2 + 1e-3 for _ in range(copies)]
    B, Bs = Bc[0], Bsc[0]
    fname = f'N={N},K={K},device_name={name},dtype=fp8_w8a8,block_shape=[32, 32].json'
    rpath = os.path.join(A.out, f'report-N{N}-K{K}{"-large" if not A.small else ""}.json')
    if A.skip_done and os.path.exists(rpath):
        print(json.dumps(dict(N=N, K=K, skipped='report exists')), flush=True)
        continue
    result, report = {}, []
    prev = []
    if A.merge and os.path.exists(os.path.join(A.out, fname)):
        result = json.load(open(os.path.join(A.out, fname)))
    for m in A.small + ([] if A.small_only else A.large):
        Aq = (torch.randn(m, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        As = torch.rand(m, K // 32, device=dev) * 1e-2 + 1e-3
        ref = run(Aq, B, As, Bs, None).float()
        # large M is compute-bound: one weight copy, few calls (rotation only matters for decode sizes)
        cp = copies if m < 1024 else 1
        iters = max(cp, 50 if m <= 32 else (10 if m <= 2048 else 3))
        base = timed(lambda i: run(Aq, Bc[i % cp], As, Bsc[i % cp], None), iters)
        best, best_cfg = base, None
        t0 = time.time()
        times = []
        candidates = [c for _, c in sorted(prev, key=lambda x: x[0])[:16]] if (prev and m >= A.shortlist_from) else space(m)
        for cfg in candidates:
            try:
                out = run(Aq, B, As, Bs, cfg).float()
                err = ((out - ref).norm() / ref.norm().clamp_min(1e-30)).item()
                if not math.isfinite(err) or err > 1e-2:
                    continue
                t = timed(lambda i: run(Aq, Bc[i % cp], As, Bsc[i % cp], cfg), iters)
            except Exception:
                continue
            times.append((t, cfg))
            if t < best:
                best, best_cfg = t, cfg
        prev = times
        default = dict(BLOCK_SIZE_M=16 if m == 1 else 64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32, GROUP_SIZE_M=32,
                       num_warps=4, num_stages=3)
        result[str(m)] = best_cfg or default
        report.append(dict(M=m, default_us=round(base, 1), tuned_us=round(best, 1), gain=round(base / best, 2),
                           searched_s=round(time.time() - t0)))
        print(json.dumps(dict(N=N, K=K, **report[-1], cfg=best_cfg)), flush=True)
    if A.small_only:  # large M keeps exactly the current default config (prefill unchanged)
        for m in A.large:
            result[str(m)] = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32, GROUP_SIZE_M=32, num_warps=4, num_stages=3)
    with open(os.path.join(A.out, fname), 'w') as f:
        json.dump(result, f, indent=4)
    with open(rpath, 'w') as f:
        json.dump(report, f, indent=1)
