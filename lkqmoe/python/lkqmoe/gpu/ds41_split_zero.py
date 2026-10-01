"""Zero the reserved page of the SM120 FlashMLA page-split scratch (LKQMOE_DS41_SPLIT_ZERO=1).

The SM120 sparse-MLA kernels want 64-token pages, so ``flash_mla_sm120._split_kv_pages_to_64`` copies the
pages a call references from the 128/256-token pools into a persistent scratch allocated with
``torch.empty``. The kernels send every invalid (-1) top-k slot to slot 0 and mask its score, but still
multiply its value row by the zero probability. Slot 0 sits in the reserved page 0, which no token
references, so that scratch page is never copied and keeps whatever the allocator left there. When that
garbage holds NaN or Inf, 0 * NaN poisons every row with fewer valid candidates than the top-k: for
DeepSeek V4.1 the first ~1023 tokens of a prompt in the ratio-2 layers (index_topk 512). Whether the
garbage is finite depends on where the scratch lands, which moved with the KV pool size: 786432 was
clean, 917504 and 1048576 were not, and prompts whose content sits in the first ~1K tokens lost it.

This zeroes the scratch pages that mirror source page 0 (the first 4, enough for 256-token source pages)
whenever a scratch buffer is (re)allocated. Page 0 is never a copy target, so it stays zero. Values the
kernels read for valid slots are unchanged.
"""
import importlib.abc
import sys

_TARGET = 'sglang.kernels.ops.attention.flash_mla_sm120'
_ZERO_PAGES = 4  # 256-token source page / 64-token scratch page


def _patch(module):
    if getattr(module, '_lkqmoe_split_zero', False):
        return
    original = module._split_kv_pages_to_64

    def split(kv_u8, src_pbs, touched_indices=None, buffer_tag='primary'):
        from sglang.srt.runtime_context import get_resources
        buffers = get_resources().buffers
        key = f'flash_mla_sm120_split:{kv_u8.device}:{buffer_tag}'
        before = buffers.get(key)
        out = original(kv_u8, src_pbs, touched_indices=touched_indices, buffer_tag=buffer_tag)
        buf = buffers.get(key)
        if buf is not None and buf is not before:
            buf[:_ZERO_PAGES].zero_()
            print(f'[lkqmoe] SM120 split scratch {buffer_tag!r}: {buf.shape[0]} pages, reserved page zeroed',
                  file=sys.stderr, flush=True)
        return out

    module._split_kv_pages_to_64 = split
    module._lkqmoe_split_zero = True
    print('[lkqmoe] SM120 split-scratch zero-page fix installed', file=sys.stderr, flush=True)


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
