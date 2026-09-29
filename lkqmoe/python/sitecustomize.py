"""lkqmoe startup hook for the DeepSeek V4.1 NVFP4 pipeline; everything is enabled by the launcher."""
import os
if os.environ.get('LKQMOE_MAIN_CPUS'):
    # Keep Python/scheduler threads off the pinned CPU expert workers; the workers and the
    # mailbox dispatcher set their own affinity.
    cpus=set()
    for part in os.environ['LKQMOE_MAIN_CPUS'].split(','):
        lo,_,hi=part.partition('-');cpus.update(range(int(lo),int(hi or lo)+1))
    os.sched_setaffinity(0,cpus&os.sched_getaffinity(0) or os.sched_getaffinity(0))
if os.environ.get('LKQMOE_MODE') in ('shadow','replace','standalone'):
    from lkqmoe.integration import install
    install()
if os.environ.get('LKQMOE_DS41_WO_A_TRITON') == '1':
    from lkqmoe.gpu.ds41_wo_a import install as install_ds41_wo_a
    install_ds41_wo_a()
if os.environ.get('LKQMOE_FP8_CONFIGS') == '1':
    from lkqmoe.gpu.fp8_configs import install as install_fp8_configs
    install_fp8_configs()
if os.environ.get('LKQMOE_CHAIN_SITECUSTOMIZE') == '1':
    # Python imports only the first sitecustomize on sys.path; run the next one (the DSpark
    # runtime overlay, which in turn loads the Engram adapter).
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
