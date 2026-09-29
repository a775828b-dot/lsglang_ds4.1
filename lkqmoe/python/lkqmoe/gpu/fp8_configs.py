"""Tuned configs for sglang's Triton W8A8 block-FP8 GEMM, kept in this repository.

sglang looks up ``configs/N=..,K=..,device_name=..,dtype=fp8_w8a8,block_shape=[..].json`` next to
``fp8_kernel.py`` and otherwise runs one default tile for every shape. The DeepSeek V4.1 FP8
projections (block 32x32; the DSpark draft's attention and the shared experts) have no file for
SM120, and the default tile is 1.5-4.6x slower than a tuned one at decode sizes
(bench/ds41_fp8_tune.py). This hook reads files from LKQMOE_FP8_CONFIG_DIR (default: the
``fp8_configs`` directory beside this module) first and falls back to sglang's own lookup, so
the frozen source tree stays untouched. Enabled with LKQMOE_FP8_CONFIGS=1.
"""
import functools
import importlib.abc
import json
import os
import sys

_TARGET = 'sglang.kernels.ops.quantization.fp8_kernel'
_DIR = os.environ.get('LKQMOE_FP8_CONFIG_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp8_configs')


def _patch(module):
    if getattr(module, '_lkqmoe_fp8_configs', False):
        return
    original = module.get_w8a8_block_fp8_configs

    @functools.lru_cache
    def lookup(N, K, block_n, block_k):
        import torch
        if torch._dynamo.is_compiling():
            return original(N, K, block_n, block_k)
        device = torch.cuda.get_device_name().replace(' ', '_')
        name = f'N={N},K={K},device_name={device},dtype=fp8_w8a8,block_shape=[{block_n}, {block_k}].json'
        path = os.path.join(_DIR, name)
        if not os.path.exists(path):
            return original(N, K, block_n, block_k)
        with open(path) as f:
            configs = {int(k): v for k, v in json.load(f).items()}
        for cfg in configs.values():  # the kernel steps scales per BLOCK_SIZE_K, which must cover block_k
            if cfg['BLOCK_SIZE_K'] < block_k or cfg['BLOCK_SIZE_K'] % block_k:
                return original(N, K, block_n, block_k)
        print(f'[lkqmoe] W8A8 block FP8 config {name} from {_DIR}', file=sys.stderr, flush=True)
        return configs

    module.get_w8a8_block_fp8_configs = lookup
    module._lkqmoe_fp8_configs = True


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
