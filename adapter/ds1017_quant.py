"""1017 reference-compatible 32-wide UE8M0 FP8 activation quant/dequant."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _round(X,Y,N:tl.constexpr,B:tl.constexpr=32):
    j=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+j,j<N,0).to(tl.float32)
    a=tl.maximum(tl.max(tl.abs(x),0),1.e-4)*(1.0/448.0)
    bits=a.to(tl.uint32,bitcast=True)
    exponent=(bits>>23)+((bits&0x7fffff)!=0).to(tl.uint32)
    scale=(exponent<<23).to(tl.float32,bitcast=True)
    q=tl.minimum(tl.maximum(x/scale,-448.0),448.0).to(tl.float8e4nv)
    tl.store(Y+j,q.to(tl.float32)*scale,j<N)

def quant_dequant(x):
    if x.dtype!=torch.bfloat16 or x.shape[-1]%32:
        raise ValueError('1017 wo_a expects BF16 and 32-wide groups')
    x=x.contiguous();y=x
    _round[(tr.cdiv(x.numel(),32),)](x,y,x.numel(),num_warps=1)
    return y
