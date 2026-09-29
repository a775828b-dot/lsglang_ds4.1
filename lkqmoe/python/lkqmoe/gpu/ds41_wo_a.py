"""DeepSeek V4 decode: route the bf16 attention ``wo_a`` low-rank to the Triton grouped einsum.

The NVIDIA ModelOpt checkpoint dequantizes ``wo_a`` to bf16 at load, and decode then runs
``torch.einsum`` (cuBLAS picks an sm80 WMMA batched GEMM). The repository's own Triton
grouped einsum (``deepseek_v4_wo_a_einsum.wo_a_bf16_einsum``, written for Ampere) is ~42%
faster on SM120 for T <= 16 tokens (verify of 1-2 requests) with identical error; for larger
T cuBLAS is as fast or faster, so it stays. Enabled with LKQMOE_DS41_WO_A_TRITON=1.
"""
import importlib.abc
import importlib.machinery
import os
import sys

_TARGET = 'sglang.srt.models.deepseek_v4'
_MAX_T = int(os.environ.get('LKQMOE_DS41_WO_A_MAX_T', '16'))


def _patch(module):
    if getattr(module, '_lkqmoe_wo_a_patched', False):
        return
    original = module._apply_wo_a_bf16_matmul

    def apply(o, wo_a, is_decode, fuse_inv_rope=False, freqs_cis=None, positions=None):
        import torch
        # Token count, not is_decode: target verify (DSpark, 6 tokens per request) runs in
        # TARGET_VERIFY mode, where is_decode is False, and is the hot path here.
        if (not fuse_inv_rope and o.dim() == 3 and 0 < o.shape[0] <= _MAX_T
                and o.dtype == torch.bfloat16 and wo_a.dtype == torch.bfloat16 and o.is_cuda):
            from sglang.srt.models.deepseek_v4_wo_a_einsum import wo_a_bf16_einsum
            return wo_a_bf16_einsum(o, wo_a, is_decode=True)
        return original(o, wo_a, is_decode, fuse_inv_rope=fuse_inv_rope, freqs_cis=freqs_cis, positions=positions)

    module._apply_wo_a_bf16_matmul = apply
    module._lkqmoe_wo_a_patched = True
    print('[lkqmoe] DeepSeek V4 wo_a decode -> Triton grouped einsum (T <= %d)' % _MAX_T, file=sys.stderr, flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, delegate):
        self.delegate = delegate

    def create_module(self, spec):
        fn = getattr(self.delegate, 'create_module', None)
        return fn(spec) if fn else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
                return spec
        return None


def install():
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
