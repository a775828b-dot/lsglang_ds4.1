"""DeepSeek V4 attention ``wo_a`` low-rank: Triton grouped einsum for decode, optionally on FP8 weights.

1. LKQMOE_DS41_WO_A_TRITON=1: the NVIDIA ModelOpt checkpoint dequantizes ``wo_a`` to bf16 at load, and
   decode then runs ``torch.einsum`` (cuBLAS picks an sm80 WMMA batched GEMM). The repository's own Triton
   grouped einsum (``deepseek_v4_wo_a_einsum.wo_a_bf16_einsum``) is ~42% faster on SM120 for T <= 16 tokens
   (the DSpark verify) with identical error; for larger T cuBLAS is as fast or faster, so it stays.

2. LKQMOE_DS41_WO_A_FP8=1 (needs 1): keep ``wo_a`` in its checkpoint form, FP8 E4M3 with E8M0 32x32 block
   scales, instead of the bf16 copy (67 -> 33.5 MB per layer, ~1.3 GB for the 40 target layers). The E8M0
   scales are powers of two, so the load-time dequantization is exact and is inverted exactly; every layer is
   checked (dequantized FP8 == the loaded bf16 bit for bit, and the FP8 einsum within 1e-3 of the bf16 einsum
   on a random input) before its bf16 copy is dropped. The FP8 kernel feeds the dot the same bf16 weights; its
   output differs from the bf16 kernel by 1 bf16 ulp on ~0.01% of elements (accumulation order), with the same
   error against an fp64 reference (1.65e-3). Decode runs an FP8 variant of the same Triton kernel that turns
   each weight tile into the same bf16 values before the dot; prefill (T > 16) dequantizes the layer into a
   reused bf16 scratch and keeps the cuBLAS einsum. Only the target model is converted (the DSpark draft
   loads through its own class and keeps bf16).
"""
import importlib.abc
import os
import re
import sys

_TARGET = 'sglang.srt.models.deepseek_v4'
_MAX_T = int(os.environ.get('LKQMOE_DS41_WO_A_MAX_T', '16'))
_FP8 = os.environ.get('LKQMOE_DS41_WO_A_FP8') == '1'
_scales = {}      # FP8 weight data_ptr -> fp32 block scales [rows/32, cols/32]
_scratch = {}     # device -> bf16 scratch for prefill dequantization
_capture = None   # {layer index: E8M0 scale} while the target model loads


def _log(msg):
    print(f'[lkqmoe] {msg}', file=sys.stderr, flush=True)


_kernels = {}


def _get_kernels():
    if _kernels:
        return _kernels
    import triton
    import triton.language as tl

    @triton.jit
    def fp8_einsum(o_ptr, wo_ptr, s_ptr, out_ptr,
                   num_tokens: tl.constexpr, num_groups: tl.constexpr, out_rank: tl.constexpr,
                   hidden_size: tl.constexpr,
                   o_stride_t, o_stride_g, o_stride_h, wo_stride_g, wo_stride_r, wo_stride_h, s_stride,
                   out_stride_t, out_stride_g, out_stride_r,
                   SCALE_BN: tl.constexpr, SCALE_BK: tl.constexpr,
                   BLOCK_T: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_H: tl.constexpr):
        # Same tiling and dot as _wo_a_bf16_einsum_kernel; only the weight tile is formed from FP8 x scale.
        t_block = tl.program_id(0)
        r_block = tl.program_id(1)
        g = tl.program_id(2)
        t_offs = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        r_offs = r_block * BLOCK_R + tl.arange(0, BLOCK_R)
        h_offs = tl.arange(0, BLOCK_H)
        t_mask = t_offs < num_tokens
        r_mask = r_offs < out_rank
        s_row = (g * out_rank + r_offs) // SCALE_BN
        accum = tl.zeros((BLOCK_T, BLOCK_R), tl.float32)
        for h_start in range(0, hidden_size, BLOCK_H):
            h = h_start + h_offs
            h_mask = h < hidden_size
            a = tl.load(o_ptr + t_offs[:, None] * o_stride_t + g * o_stride_g + h[None, :] * o_stride_h,
                        mask=t_mask[:, None] & h_mask[None, :], other=0.0)
            w8 = tl.load(wo_ptr + g * wo_stride_g + r_offs[:, None] * wo_stride_r + h[None, :] * wo_stride_h,
                         mask=r_mask[:, None] & h_mask[None, :], other=0.0)
            # one scale per 32 columns: load [BLOCK_R, BLOCK_H // SCALE_BK] and apply it to 32-wide slices
            k_offs = h_start // SCALE_BK + tl.arange(0, BLOCK_H // SCALE_BK)
            s = tl.load(s_ptr + s_row[:, None] * s_stride + k_offs[None, :],
                        mask=r_mask[:, None] & (k_offs * SCALE_BK < hidden_size)[None, :], other=0.0)
            wf = tl.reshape(w8.to(tl.float32), (BLOCK_R, BLOCK_H // SCALE_BK, SCALE_BK))
            b = tl.reshape(wf * s[:, :, None], (BLOCK_R, BLOCK_H)).to(tl.bfloat16)
            accum = tl.dot(a, tl.trans(b), acc=accum, input_precision="tf32")
        tl.store(out_ptr + t_offs[:, None] * out_stride_t + g * out_stride_g + r_offs[None, :] * out_stride_r,
                 accum, mask=t_mask[:, None] & r_mask[None, :])

    @triton.jit
    def dequant(w_ptr, s_ptr, out_ptr, rows, cols, s_stride,
                SCALE_BN: tl.constexpr, SCALE_BK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        r = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (r < rows)[:, None] & (c < cols)[None, :]
        w = tl.load(w_ptr + r[:, None] * cols + c[None, :], mask=mask, other=0.0)
        s = tl.load(s_ptr + (r // SCALE_BN)[:, None] * s_stride + (c // SCALE_BK)[None, :], mask=mask, other=0.0)
        tl.store(out_ptr + r[:, None] * cols + c[None, :], (w.to(tl.float32) * s).to(tl.bfloat16), mask=mask)

    _kernels['einsum'] = fp8_einsum
    _kernels['dequant'] = dequant
    return _kernels


def _fp8_einsum(o, wo8, scale):
    import torch
    import triton
    T, G, D = o.shape
    _, R, _ = wo8.shape
    out = torch.empty((T, G, R), dtype=o.dtype, device=o.device)
    bn = G * R // scale.shape[0]
    bk = D // scale.shape[1]
    grid = (triton.cdiv(T, 16), triton.cdiv(R, 128), G)
    _get_kernels()['einsum'][grid](
        o, wo8, scale, out, T, G, R, D,
        o.stride(0), o.stride(1), o.stride(2), wo8.stride(0), wo8.stride(1), wo8.stride(2), scale.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        SCALE_BN=bn, SCALE_BK=bk, BLOCK_T=16, BLOCK_R=128, BLOCK_H=128, num_warps=4, num_stages=3)
    return out


def _dequant_into(wo8_2d, scale, out_2d):
    import triton
    rows, cols = wo8_2d.shape
    grid = (triton.cdiv(rows, 64), triton.cdiv(cols, 128))
    _get_kernels()['dequant'][grid](wo8_2d, scale, out_2d, rows, cols, scale.stride(0),
                                    SCALE_BN=rows // scale.shape[0], SCALE_BK=cols // scale.shape[1],
                                    BLOCK_M=64, BLOCK_N=128, num_warps=4)
    return out_2d


def _convert(model, captured):
    """Replace each captured layer's bf16 wo_a by its exact FP8 form (checked bit for bit)."""
    import torch
    from sglang.srt.models.deepseek_v4_wo_a_einsum import wo_a_bf16_einsum
    done = skipped = saved = 0
    for name, mod in model.named_modules():
        if not (hasattr(mod, 'wo_a') and hasattr(mod, 'o_lora_rank') and hasattr(mod, 'n_local_groups')):
            continue
        m = re.search(r'layers\.(\d+)\.', name + '.')
        if not m or int(m.group(1)) not in captured:
            continue
        w = getattr(mod.wo_a, 'weight', None)
        if w is None or w.dtype != torch.bfloat16 or not w.is_cuda:
            continue
        s = captured[int(m.group(1))].to(device=w.device, dtype=torch.float32).contiguous()
        sn, sk = s.shape
        bn, bk = w.shape[0] // sn, w.shape[1] // sk
        blocks = w.float().view(sn, bn, sk, bk)
        fp8 = (blocks / s[:, None, :, None]).view_as(w).to(torch.float8_e4m3fn)
        back = (fp8.float().view(sn, bn, sk, bk) * s[:, None, :, None]).view_as(w).to(torch.bfloat16)
        ok = torch.equal(back, w)
        if ok:
            G, R = mod.n_local_groups, mod.o_lora_rank
            o = torch.randn(6, G, w.shape[1], device=w.device, dtype=torch.bfloat16)
            ref = wo_a_bf16_einsum(o, w.view(G, R, -1), is_decode=True)
            got = _fp8_einsum(o, fp8.view(G, R, -1), s)
            # same weights and dot; only the compiler's accumulation order may differ (1 bf16 ulp on ~0.01% of
            # outputs, identical error against an fp64 reference), so compare within 1e-3 relative
            ok = ((ref.float() - got.float()).norm() / ref.float().norm().clamp_min(1e-30)).item() <= 1e-3
        del blocks, back
        if not ok:
            skipped += 1
            continue
        mod.wo_a.weight = torch.nn.Parameter(fp8, requires_grad=False)
        _scales[fp8.data_ptr()] = s
        saved += w.numel()
        done += 1
    torch.cuda.empty_cache()
    _log(f'wo_a FP8: {done} layers converted (checked), {skipped} kept bf16, '
         f'{saved / 2**30:.2f} GiB of bf16 weights freed')


def _patch(module):
    if getattr(module, '_lkqmoe_wo_a_patched', False):
        return
    original = module._apply_wo_a_bf16_matmul

    def apply(o, wo_a, is_decode, fuse_inv_rope=False, freqs_cis=None, positions=None):
        import torch
        if wo_a.dtype == torch.float8_e4m3fn:
            scale = _scales[wo_a.data_ptr()]
            if not fuse_inv_rope and o.dim() == 3 and 0 < o.shape[0] <= _MAX_T and o.dtype == torch.bfloat16:
                return _fp8_einsum(o, wo_a, scale)
            key = (wo_a.device, wo_a.numel())
            buf = _scratch.get(key)
            if buf is None:
                buf = _scratch[key] = torch.empty(wo_a.shape, dtype=torch.bfloat16, device=wo_a.device)
            _dequant_into(wo_a.reshape(-1, wo_a.shape[-1]), scale, buf.view(-1, wo_a.shape[-1]))
            wo_a = buf
        # Token count, not is_decode: target verify (DSpark, 6 tokens per request) runs in
        # TARGET_VERIFY mode, where is_decode is False, and is the hot path here.
        if (not fuse_inv_rope and o.dim() == 3 and 0 < o.shape[0] <= _MAX_T
                and o.dtype == torch.bfloat16 and wo_a.dtype == torch.bfloat16 and o.is_cuda):
            from sglang.srt.models.deepseek_v4_wo_a_einsum import wo_a_bf16_einsum
            return wo_a_bf16_einsum(o, wo_a, is_decode=True)
        return original(o, wo_a, is_decode, fuse_inv_rope=fuse_inv_rope, freqs_cis=freqs_cis, positions=positions)

    module._apply_wo_a_bf16_matmul = apply
    if _FP8:
        streaming = module._dequant_fp8_wo_a_streaming

        def tapped(weights):
            for name, tensor in weights:
                if _capture is not None and name.endswith('.wo_a.scale'):
                    m = re.search(r'(?:^|\.)layers\.(\d+)\.attn\.wo_a\.scale$', name)
                    if m:
                        _capture[int(m.group(1))] = tensor.detach().clone()
                yield name, tensor

        def streaming_with_capture(weights):
            return streaming(tapped(weights) if _capture is not None else weights)

        module._dequant_fp8_wo_a_streaming = streaming_with_capture
        cls = module.DeepseekV4ForCausalLM
        load_weights = cls.load_weights

        def load_weights_fp8(self, *args, **kwargs):
            global _capture
            _capture = {}
            try:
                result = load_weights(self, *args, **kwargs)
                captured, _capture = _capture, None
                if captured:
                    _convert(self, captured)
                return result
            finally:
                _capture = None

        cls.load_weights = load_weights_fp8
    module._lkqmoe_wo_a_patched = True
    _log('DeepSeek V4 wo_a decode -> Triton grouped einsum (T <= %d)%s' % (_MAX_T, ', FP8 weights' if _FP8 else ''))


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
