"""Explicitly enabled startup hook; otherwise no imports or behavior changes."""
import os
if os.environ.get('LKQMOE_MAIN_CPUS'):
    # Keep Python/scheduler threads off the pinned CPU expert workers (0..63 on the
    # validated host); workers and the mailbox dispatcher set their own affinity.
    cpus=set()
    for part in os.environ['LKQMOE_MAIN_CPUS'].split(','):
        lo,_,hi=part.partition('-');cpus.update(range(int(lo),int(hi or lo)+1))
    os.sched_setaffinity(0,cpus&os.sched_getaffinity(0) or os.sched_getaffinity(0))
if os.environ.get('LKQMOE_MODE') in ('shadow','replace','standalone'):
    from lkqmoe.integration import install
    install()
if os.environ.get('LKQMOE_SMALL_M_GEMM') == '1':
    from lkqmoe.gpu.small_gemm import install as install_small_gemm
    install_small_gemm()
if os.environ.get('LKQMOE_MTP_QUANT') == '1':
    from lkqmoe.gpu.mtp_quant import install as install_mtp_quant
    install_mtp_quant()
if os.environ.get('LKQMOE_MTP_ROUTER_GUARD') == '1':
    from lkqmoe.gpu.router_dependency import install as install_router
    install_router()
if any(os.environ.get(key)=='1' for key in ['LKQMOE_EXPERIMENTAL_FP8_PROJECTION','LKQMOE_EXPERIMENTAL_HC_MIX','LKQMOE_EXPERIMENTAL_PREFILL_PREFETCH']):
    from lkqmoe.gpu.startup import install as install_projection
    install_projection()
if os.environ.get('LKQMOE_DS41_WO_A_TRITON') == '1':
    from lkqmoe.gpu.ds41_wo_a import install as install_ds41_wo_a
    install_ds41_wo_a()
if os.environ.get('LKQMOE_FP8_CONFIGS') == '1':
    from lkqmoe.gpu.fp8_configs import install as install_fp8_configs
    install_fp8_configs()
if os.environ.get('LKQMOE_CHAIN_SITECUSTOMIZE') == '1':
    # Python imports only the first sitecustomize on sys.path. Pipelines that ship their
    # own (e.g. the DeepSeek V4.1 runtime overlay) put this directory first and chain on.
    import runpy, sys
    here = os.path.dirname(os.path.abspath(__file__))
    after = False
    for entry in sys.path:
        directory = os.path.abspath(entry or os.getcwd())
        if directory == here:
            after = True
        elif after and os.path.isfile(os.path.join(directory, 'sitecustomize.py')):
            runpy.run_path(os.path.join(directory, 'sitecustomize.py'), run_name='sitecustomize_chained')
            break
# Hooks that must wrap modules the runtime overlay provides go after the chain above, so their finders sit
# in front of the overlay's.
if int(os.environ.get('LKQMOE_DS41_DENSE_INDEXER_MIB', '0') or 0) > 0:
    from lkqmoe.gpu.ds41_dense_indexer import install as install_ds41_dense_indexer
    install_ds41_dense_indexer()
if os.environ.get('LKQMOE_DS41_C128_PROBE') == '1':
    from lkqmoe.gpu.ds41_c128 import install as install_ds41_c128
    install_ds41_c128()
if os.environ.get("LKQMOE_DS41_TRACE") == "1":
    from lkqmoe.gpu.ds41_trace import install as install_ds41_trace
    install_ds41_trace()
if os.environ.get("LKQMOE_DS41_SPLIT_ZERO") == "1":
    from lkqmoe.gpu.ds41_split_zero import install as install_ds41_split_zero
    install_ds41_split_zero()
