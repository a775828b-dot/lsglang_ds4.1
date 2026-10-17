"""Hand each CPU FP4 MoE layer to lkqmoe as soon as its experts are loaded, then drop its staging.

The FP4 expert parameters of CPU layers are staged in mappings of deleted scratch files
(SGLANG_LK_MOE_NVME_STAGING, utils/lk_moe_nvme_alloc.py). sglang processes MoE layers only after
the whole checkpoint is loaded, so all ~288 GB of staging is dirty at once, more than the free RAM:
the kernel writes it back to the scratch disk and the loader threads wait on that writeback
(DeepSeek V4.1 1017: ~20 min of load_weight on a ~216 MB/s system disk).

Here a background thread processes the layers in construction (layer) order as soon as each one
has all its expert tensors: FusedMoE.process_weights_after_loading (lkqmoe copies the weights into
its own NUMA shards), then the staging tensors are replaced by empty ones so their mappings close.
The scratch files are already unlinked, so their dirty pages are dropped instead of written back.
Same calls and arguments as the loader's own post-load loop, which skips layers handled here
(process_lk_moe_already_called). A layer whose expert count never completes is left to that loop.
Enabled by DS41_EARLY_LK_MOE=1.
"""
import logging
import os
import threading
import time

import torch

logger = logging.getLogger(__name__)
STAGED = ('w13_weight', 'w2_weight', 'w13_weight_scale_inv', 'w2_weight_scale_inv')


def install(module):
    FusedMoE = module.FusedMoE
    if getattr(FusedMoE, '_early_lk_installed', False):
        return
    FusedMoE._early_lk_installed = True
    cond = threading.Condition()
    state = dict(layers=[], next=0, worker=None)
    orig_init = FusedMoE.__init__
    orig_loader = FusedMoE.weight_loader
    orig_process = FusedMoE.process_weights_after_loading

    def eligible(layer):
        qm = getattr(layer, 'quant_method', None)
        # MXFP4 (Mxfp4FlashinferCutlassMoEMethod) or NVFP4 experts staged on scratch; others keep the post-load pass
        return (os.environ.get('SGLANG_LK_MOE_NVME_STAGING') == '1' and not getattr(layer, 'is_gpu_resident_layer', True)
                and (getattr(qm, 'is_fp4_expert', False) or type(qm).__name__.startswith('Mxfp4'))
                and all(isinstance(getattr(layer, n, None), torch.Tensor) for n in STAGED))

    def release(layer):
        for name in STAGED:
            p = getattr(layer, name, None)
            if isinstance(p, torch.Tensor):
                p.data = torch.empty(0, dtype=p.dtype)   # closes the scratch mapping (params_dict still holds p)

    def process(layer):
        with layer._early_lock:
            if layer._early_done:
                return False
            orig_process(layer)
            layer._early_done = True
        release(layer)
        layer.clean_weights_after_loading()
        setattr(layer, 'process_lk_moe_already_called', True)
        return True

    def worker():
        n, t0 = 0, time.time()
        while True:
            with cond:
                while True:
                    i, layers = state['next'], state['layers']
                    if i < len(layers) and layers[i]._early_complete:
                        layer = layers[i]
                        break
                    if not cond.wait(timeout=600):
                        logger.info('early lk_moe: %d layers processed during loading in %.0f s', n, time.time() - t0)
                        return
            try:
                n += process(layer)
                if n in (1, 10, 20, 30, 40):
                    logger.info('early lk_moe: %d layers processed (last layer %s) after %.0f s', n, getattr(layer, 'layer_id', '?'), time.time() - t0)
            except Exception:
                logger.exception('early lk_moe: layer %s failed; left to the post-load pass', getattr(layer, 'layer_id', '?'))
            with cond:
                state['next'] += 1
                cond.notify_all()

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if eligible(self):
            self._early_lock = threading.Lock()
            self._early_done = self._early_complete = False
            self._early_seen = set()
            self._early_expected = 6 * self.moe_runner_config.num_local_experts   # w1/w3/w2 weight + scale per expert
            with cond:
                state['layers'].append(self)
                if state['worker'] is None:
                    state['worker'] = threading.Thread(target=worker, name='early-lk-moe', daemon=True)
                    state['worker'].start()

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, *args, **kwargs):
        result = orig_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, *args, **kwargs)
        seen = getattr(self, '_early_seen', None)
        if seen is not None and expert_id is not None and not self._early_complete \
                and any(param is getattr(self, n, None) for n in STAGED):
            with cond:
                seen.add((weight_name, shard_id, expert_id))
                if len(seen) >= self._early_expected:
                    self._early_complete = True
                    cond.notify_all()
        return result

    def process_weights_after_loading(self):
        lock = getattr(self, '_early_lock', None)
        if lock is None:
            return orig_process(self)
        with lock:
            if self._early_done:
                return
            orig_process(self)
            self._early_done = True

    FusedMoE.__init__ = __init__
    FusedMoE.weight_loader = weight_loader
    FusedMoE.process_weights_after_loading = process_weights_after_loading
    logger.info('early lk_moe processing installed (DS41_EARLY_LK_MOE=1)')
