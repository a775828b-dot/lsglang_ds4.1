"""Bounded exact file-backed replacement for EngramEmbedding's owned-row gather.

The original hash, gating, projections, TP all-reduce and model remain unchanged.
Native callback executes IO inside CUDA graphs without calling the CUDA API.
"""
import ctypes as C
import glob
import json
import logging
import os
from pathlib import Path
import struct
import threading
import time
import urllib.request

_scale_load_lock = threading.Lock()

import torch

P, U = C.c_void_p, C.c_uint64
_lib = C.CDLL(str(Path(__file__).with_name('librow_store.so')))
_lib.row_store_open.argtypes = [C.c_char_p, U, U, U, U]
_lib.row_store_open.restype = P
_lib.row_store_range.argtypes = [P, U, U]
_lib.row_store_stats.argtypes = [P, C.POINTER(U)]
_lib.row_store_stats.restype = None
_lib.row_store_io_stats.argtypes = [P, C.POINTER(U)]
_lib.row_store_preload_scales.argtypes = [P]
_lib.row_store_preload_scales.restype = C.c_int
_lib.row_store_timing.argtypes = [P, C.POINTER(U)]
_lib.row_store_timing.restype = None

# 2026-10-19 (DS41_ENGRAM_PREFETCH=1): a decode / verify step needs the rows of Engram layers 1
# and 14 only when it reaches them, but their hash ids exist right after hashing. Each layer's
# lookup (ids D2H, host callback, rows H2D, gather) then runs on its own side stream from there,
# overlapping layer 0 (and layers 2-13 for layer 14) instead of stalling the main stream at the
# layer. Same rows, same kernels; the wkv projection stays on the main stream (its GEMM backend
# may share a workspace across streams). Prefill keeps the inline lookup.
_PREFETCH = os.environ.get('DS41_ENGRAM_PREFETCH', '0') == '1'
_REGISTRY = {}   # model layer id -> EngramEmbedding of this rank

class Work(C.Structure):
    _fields_ = [('store', P), ('ids', P), ('weights', P), ('scales', P), ('count', U)]

def _find_cudart():
    patterns = [
        '/usr/local/cuda*/targets/x86_64-linux/lib/libcudart.so*',
        '/usr/local/lib/python*/dist-packages/nvidia/cuda_runtime/lib/libcudart.so*',
        '/usr/local/lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*',
    ]
    try:
        import nvidia.cuda_runtime as _ncr

        patterns.append(str(Path(_ncr.__file__).parent / 'lib' / 'libcudart.so*'))
    except Exception:
        pass
    try:
        import torch as _torch

        root = Path(_torch.__file__).resolve().parent.parent
        # CUDA 13 wheels ship under nvidia/cu13/lib; older under
        # nvidia/cuda_runtime/lib.
        patterns.append(str(root / 'nvidia' / '*' / 'lib' / 'libcudart.so*'))
    except Exception:
        pass
    patterns.append('/usr/lib/x86_64-linux-gnu/libcudart.so*')
    for pattern in patterns:
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    raise OSError('libcudart.so not found; searched: %s' % patterns)


_cuda = C.CDLL(_find_cudart())
_cuda.cudaLaunchHostFunc.argtypes = [P, P, P]
_cuda.cudaLaunchHostFunc.restype = C.c_int

def install(module):
    cls = module.EngramEmbedding
    def init(self, num_embeddings, dim, layer_id):
        torch.nn.Module.__init__(self)
        assert dim == 256
        self.dim, self.tp_size = dim, module.get_parallel().tp_size
        rank = module.get_parallel().tp_rank
        self.row_start = num_embeddings * rank // self.tp_size
        end = num_embeddings * (rank + 1) // self.tp_size
        self.rows, self.host_table = end - self.row_start, None
        root = Path(os.environ['DSV41_SOURCE'])
        index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        prefix = f'layers.{layer_id}.engram.embed.'
        filename = root/index[prefix+'weight']
        assert index[prefix+'weight'] == index[prefix+'scale']
        with filename.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(length))
        w, s = header[prefix+'weight'], header[prefix+'scale']
        assert w['shape'] == [num_embeddings, 256] and w['dtype'] == 'F8_E4M3'
        assert s['shape'] == [num_embeddings, 8] and s['dtype'] == 'F8_E8M0'
        budget = int(float(os.getenv('DSV41_CACHE_GIB', '64')) * 2**30) // (2*self.tp_size)
        self._store = _lib.row_store_open(str(filename).encode(), num_embeddings,
            8+length+w['data_offsets'][0], 8+length+s['data_offsets'][0], budget)
        if not self._store:
            raise RuntimeError(f'Could not open Engram backing shard: {filename}')
        _lib.row_store_range(self._store, self.row_start, end)
        stats_dir = os.getenv('DSV41_STATS_DIR')
        if stats_dir:
            destination = Path(stats_dir)
            destination.mkdir(parents=True, exist_ok=True)
            def report_stats():
                scale_error = None
                while True:
                    io_values = (U * 3)()
                    _lib.row_store_io_stats(self._store, io_values)
                    trigger = os.getenv('DSV41_SCALE_RAM_TRIGGER')
                    if trigger and Path(trigger).exists() and not io_values[2] and not scale_error:
                        with _scale_load_lock:
                            # Wait for the model and CUDA graphs before allocating scale RAM.
                            try:
                                with urllib.request.urlopen(os.environ.get('DSV41_HEALTH_URL', 'http://127.0.0.1:39500/health'), timeout=5) as response:
                                    ready = response.status == 200
                            except Exception:
                                ready = False
                            if ready:
                                need = self.rows * 8   # the scales of this rank's rows only
                                available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
                                cg = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
                                cgroup = Path('/sys/fs/cgroup') / cg.lstrip('/')
                                current = int((cgroup / 'memory.current').read_text())
                                limit = (cgroup / 'memory.max').read_text().strip()
                                enough = available > need + 8 * 2**30 and (limit == 'max' or current + need + 3 * 2**30 < int(limit))
                                if enough:
                                    if _lib.row_store_preload_scales(self._store):
                                        scale_error = 'scale preload failed; continuing exact NVMe lookup'
                                    _lib.row_store_io_stats(self._store, io_values)
                    values = (U * 4)()
                    _lib.row_store_stats(self._store, values)
                    timing = (U * 3)()
                    _lib.row_store_timing(self._store, timing)
                    record = dict(layer=layer_id, rank=rank, pid=os.getpid(), mode=os.getenv('OFFLOAD_MODE'),
                                  hits=values[0], misses=values[1], reads=values[2], cache_bytes=values[3], timestamp=time.time(), io_batches=io_values[0],
                                  peak_batch_pages=io_values[1], scale_ram_bytes=io_values[2], scale_error=scale_error,
                                  lookups=timing[1], lookup_ns_total=timing[0], lookup_max_ns=timing[2],
                                  prefetch=_PREFETCH)
                    # One file per TP rank (rank 0 keeps the single-GPU name): ranks sharing a name
                    # race on the rename and the loser's thread dies.
                    name = f'layer-{layer_id}' + (f'-rank{rank}' if rank else '')
                    temporary = destination / f'{name}.tmp'
                    temporary.write_text(json.dumps(record))
                    temporary.replace(destination / f'{name}.json')
                    time.sleep(5)
            threading.Thread(target=report_stats, daemon=True).start()
        self._staging = {}
        self._works = {}
        self._pending = None
        self._prefetch_stream = None
        _REGISTRY[layer_id] = self
        # The loader sees names but never allocates/copies the complete tables.
        self.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.scale = torch.nn.Parameter(torch.empty(0, dtype=torch.float8_e8m0fnu), requires_grad=False)
        def validate_weight(param, source):
            expected = w if param is self.weight else s
            if list(source.shape) != expected['shape']:
                raise ValueError('Engram checkpoint shape mismatch')
        self.weight.weight_loader = validate_weight
        self.scale.weight_loader = validate_weight
        logging.getLogger(__name__).info('Exact %s Engram layer=%s rank=%s rows=[%s,%s) cache_budget=%s',
                                        os.getenv('OFFLOAD_MODE', 'nvme'), layer_id, rank, self.row_start, end, budget)

    def prefetch(self, indices):
        """Start this layer's lookup of `indices` on the side stream (main stream, after hashing)."""
        self._pending = None
        if self._prefetch_stream is None:
            if torch.cuda.is_current_stream_capturing():
                return   # created on the eager warm-up pass before any capture
            self._prefetch_stream = torch.cuda.Stream(device=indices.device)
        side = self._prefetch_stream
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            out = lookup(self, indices)
        indices.record_stream(side)
        self._pending = (indices.data_ptr(), tuple(indices.shape), tuple(indices.stride()), out)

    def owned(self, indices):
        pending, self._pending = self._pending, None
        if pending is not None and pending[:3] == (indices.data_ptr(), tuple(indices.shape), tuple(indices.stride())):
            main = torch.cuda.current_stream()
            main.wait_stream(self._prefetch_stream)
            pending[3].record_stream(main)
            return pending[3]
        return lookup(self, indices)

    def lookup(self, indices):
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather
        count = indices.numel()
        if not count:
            return self._empty(indices)
        capacity = 1 << (count - 1).bit_length()
        key = (indices.device.index, capacity)
        if key not in self._staging:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f'Engram staging {key} must be warmed before graph capture')
            ids = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
            w = torch.empty((capacity, 256), dtype=torch.uint8, pin_memory=True)
            s = torch.empty((capacity, 8), dtype=torch.uint8, pin_memory=True)
            dw, ds = w.to(indices.device), s.to(indices.device)
            sequential = torch.arange(capacity, dtype=torch.int64, device=indices.device)
            self._staging[key] = (ids, w, s, dw, ds, sequential)
        ids, w, s, dw, ds, sequential = self._staging[key]
        work_key = (indices.device.index, count)
        if work_key not in self._works:
            self._works[work_key] = Work(self._store, ids.data_ptr(), w.data_ptr(), s.data_ptr(), count)
        work = self._works[work_key]
        ids[:count].copy_(indices.reshape(-1), non_blocking=True)
        error = _cuda.cudaLaunchHostFunc(torch.cuda.current_stream().cuda_stream,
            C.cast(_lib.row_store_lookup, P), C.addressof(work))
        if error:
            raise RuntimeError(f'CUDA Engram host callback failed: {error}')
        dw[:count].copy_(w[:count], non_blocking=True)
        ds[:count].copy_(s[:count], non_blocking=True)
        out = self._empty(indices)
        engram_gather(dw.data_ptr(), ds.data_ptr(), sequential[:count], out.view(-1, 256), 256, 32)
        return out
    cls.__init__ = init
    cls._owned_rows = owned
    cls.prefetch = prefetch

    hasher = module.EngramHasher
    hasher_init, hasher_forward = hasher.__init__, hasher.forward

    def init_hasher(self, layout, *args, **kwargs):
        hasher_init(self, layout, *args, **kwargs)
        self._ds41_layer_ids = list(layout.layer_ids)   # hash_ids[:, j] belongs to layer_ids[j]

    def forward_hasher(self, input_ids, forward_batch):
        hash_ids = hasher_forward(self, input_ids, forward_batch)
        mode = forward_batch.forward_mode
        if (_PREFETCH and hash_ids.is_cuda and hash_ids.shape[0]
                and (mode.is_decode() or mode.is_target_verify())):
            for j, layer_id in enumerate(self._ds41_layer_ids):
                embedding = _REGISTRY.get(layer_id)
                if embedding is not None and embedding.rows:
                    embedding.prefetch(hash_ids[:, j])
        return hash_ids

    hasher.__init__ = init_hasher
    hasher.forward = forward_hasher
