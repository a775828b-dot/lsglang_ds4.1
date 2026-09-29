"""Build a draft-only copy of the DeepSeek V4.1 checkpoint for the DSpark draft model.

Usage: python make_draft_slim.py SRC DST [--prefixes mtp.] [--max-shard-gib 2]
The DSpark draft is loaded from the full official checkpoint (476 GB, 48 shards), and the
loader walks every shard although the draft needs only ~10 GiB (mtp.* plus the top-level
embed/norm/head/image tensors). This writes those tensors, unchanged, into small shards with
their own index; every other file is symlinked. Output shards are written one at a time so
peak host memory stays near --max-shard-gib.
"""
import argparse, json, os
from safetensors import safe_open
from safetensors.torch import save_file

P = argparse.ArgumentParser()
P.add_argument('src'); P.add_argument('dst')
P.add_argument('--max-shard-gib', type=float, default=2.0)
A = P.parse_args()
src, dst = os.path.realpath(A.src), A.dst
os.makedirs(dst, exist_ok=True)
index = json.load(open(os.path.join(src, 'model.safetensors.index.json')))
wm = index['weight_map']
keys = [k for k in wm if k.startswith('mtp.') or not k.startswith(('layers.', 'vision', 'aligner'))]
for name in os.listdir(src):
    if name.endswith('.safetensors') or name == 'model.safetensors.index.json':
        continue
    link = os.path.join(dst, name)
    if not os.path.lexists(link):
        os.symlink(os.path.join(src, name), link)
# group by source file to read each shard once, then pack into <= max-shard-gib outputs
by_file = {}
for k in keys:
    by_file.setdefault(wm[k], []).append(k)
limit = int(A.max_shard_gib * 2**30)
out_map, batch, size, n, total = {}, {}, 0, 0, 0
def flush():
    global batch, size, n
    if not batch:
        return
    n += 1
    name = f'draft-{n:05d}.safetensors'
    save_file(batch, os.path.join(dst, name), metadata={'format': 'pt'})
    for k in batch:
        out_map[k] = name
    print(f'{name}: {len(batch)} tensors, {size / 2**30:.2f} GiB', flush=True)
    batch, size = {}, 0
for f in sorted(by_file):
    with safe_open(os.path.join(src, f), 'pt') as s:
        for k in sorted(by_file[f]):
            t = s.get_tensor(k)
            nbytes = t.numel() * t.element_size()
            if size and size + nbytes > limit:
                flush()
            batch[k] = t.contiguous(); size += nbytes; total += nbytes
flush()
json.dump({'metadata': {'total_size': total}, 'weight_map': out_map},
          open(os.path.join(dst, 'model.safetensors.index.json'), 'w'), indent=1)
print(f'{len(out_map)} tensors, {total / 2**30:.2f} GiB -> {dst}')
