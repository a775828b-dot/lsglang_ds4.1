"""Read a model's safetensors shards into page cache just ahead of the loader.

Usage: python shard_prefetch.py MODEL_DIR [--lead 4] [--threads 4] [--timeout 3600]
The loader maps shards (mmap) and faults them in small random reads; on consumer NVMe that
runs at ~150 MB/s. This reads the shards sequentially in 16 MiB blocks, in file order, but
never more than --lead files beyond the highest shard any process currently has mapped, so
page cache does not evict pages before the loader uses them (RAM is nearly full while the
MoE shards are built). Unlike sglang's --weight-loader-prefetch-checkpoints it touches only
this model directory (not e.g. the speculative draft's full source checkpoint). Exits when
all shards have been read, or on timeout.
"""
import argparse, glob, os, re, threading, time

P = argparse.ArgumentParser()
P.add_argument('model_dir')
P.add_argument('--lead', type=int, default=4)
P.add_argument('--threads', type=int, default=4)
P.add_argument('--timeout', type=float, default=3600)
P.add_argument('--idle-exit', type=float, default=30, help='exit this long after the loader unmapped all shards')
P.add_argument('--cpus', default='64-79,81-127', help='CPU affinity (keep off the expert workers)')
A = P.parse_args()
try:
    cpus = set()
    for part in A.cpus.split(','):
        lo, _, hi = part.partition('-'); cpus.update(range(int(lo), int(hi or lo) + 1))
    os.sched_setaffinity(0, cpus & os.sched_getaffinity(0) or os.sched_getaffinity(0))
except (OSError, ValueError):
    pass
files = sorted(glob.glob(os.path.join(os.path.realpath(A.model_dir), '*.safetensors')))
index = {f: i for i, f in enumerate(files)}
pattern = re.compile(re.escape(os.path.realpath(A.model_dir)) + r'/[^/\s]+\.safetensors')


state = dict(best=-1, seen=False, last_seen=time.time())


def loader_position():
    """Highest shard index mapped so far by a python/sglang process (monotonic)."""
    mapped = False
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = f.read(4096)
                if b'python' not in cmd and b'sglang' not in cmd:
                    continue
            with open(f'/proc/{pid}/maps') as f:
                for m in pattern.findall(f.read()):
                    mapped = True
                    state['best'] = max(state['best'], index.get(m, -1))
        except OSError:
            continue
    if mapped:
        state['seen'] = True; state['last_seen'] = time.time()
    return state['best']


def loader_done():
    return state['seen'] and time.time() - state['last_seen'] > A.idle_exit


lock = threading.Lock()
next_file = [0]
start = time.time()


def worker():
    buf = bytearray(16 << 20)
    while time.time() - start < A.timeout:
        with lock:
            i = next_file[0]
            if i >= len(files):
                return
            if loader_done():
                return
            if i > loader_position() + A.lead:
                wait = True
            else:
                next_file[0] += 1
                wait = False
        if wait:
            time.sleep(1.0)
            continue
        with open(files[i], 'rb', buffering=0) as f:
            while f.readinto(buf):
                pass


threads = [threading.Thread(target=worker, daemon=True) for _ in range(A.threads)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f'[shard_prefetch] {min(next_file[0], len(files))}/{len(files)} shards in {time.time() - start:.0f} s', flush=True)
