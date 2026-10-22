#!/usr/bin/env bash
# DeepSeek V4.1 Flash 1017 MXFP4 + DSpark on one RTX PRO 5000 72GB (SM120) + 2x EPYC 9334-class Genoa,
# all 24 memory channels populated (24 x 16 GiB DDR5-4800, 384 GiB) and BIOS NPS4 (8 NUMA nodes), 2026-10-22.
# Same framework, lkqmoe build and defaults as run-ds41-1017.sh (20 channels, NPS2); the difference is the CPU
# split: lkqmoe puts 14 expert threads on each of the 8 nodes (8 cores + 6 SMT siblings), and the host / loader
# CPUs are resolved from the library's own worker list instead of the NPS2 list, which overlaps 8 of them here.
# Routed experts of the CPU layers run on lkqmoe (compiled, ../lkqmoe); with LKQstrata (default) their hot experts
# also get GPU slots (lkqmoe 0.5.0 expert pool) and no target MoE layer stays whole on the GPU.
# This is the launch used for the measurements in README.md; edit the paths block first.
set -Eeuo pipefail

# ---- paths ------------------------------------------------------------------------------
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
LSGLANG=${LSGLANG:-/opt/Lsglang}                       # guqiong96/Lsglang @ 6068667 + patches/lsglang-6068667-1017.patch
VENV=${VENV:-$LSGLANG/env}                             # python 3.12, see README "Environment"
FLASHINFER_SM120=${FLASHINFER_SM120:-$VENV/flashinfer-0.6.18}                 # optional: FlashInfer 0.6.18 SM120 build (python path)
MODEL=${MODEL:-/models/deepseek-ai/DeepSeek-V4.1-Flash-1017}
DRAFT=${DRAFT:-${MODEL}-dspark}  # Extract only from the same 1017 checkpoint
LOADER_SCRATCH=${LOADER_SCRATCH:-/scratch/ds41-loader}   # NVMe, >= 320 GiB free (FP4 weight staging while loading)
RUN_DIR=${RUN_DIR:-$REPO/run}
PORT=${PORT:-39504}
HOST=${HOST:-127.0.0.1}
VERIFY_MODE=${VERIFY_MODE:-static}    # static (default) | cap-sts; variable-length verify (compact) dropped 2026-10-21
STS_CALIBRATION=${STS_CALIBRATION:-$REPO/config/dspark-sts-calib.json}

# ---- settings used for the published numbers --------------------------------------------
# 1017: all 40 target MoE layers use CPU plus GPU hot-expert slots (50 per layer at the default batch).
# Draft layers stay on GPU. Two requests share the 512K KV pool.
LKQSTRATA=${LKQSTRATA:-1}
MAIN_LAYERS=${MAIN_LAYERS:-0}
RESIDENT_START=${RESIDENT_START:-0}
ENGRAM_CACHE_GIB=${ENGRAM_CACHE_GIB:-4}
BATCH_TOKENS=${BATCH_TOKENS:-22528}    # chunked prefill size (2026-10-19; was 32768)
# A smaller prefill chunk needs less VRAM (~0.5 GiB per 1K tokens at 500K context), spent on hot-expert slots.
# Slots per batch size, each leaving >= 0.5 GiB at the 522K+1K peak on the 72 GB card (KV 512K):
# 22528 -> 50 (default), 20480 -> 52, 32768 -> 43 (measured), 24576 -> 47 (estimate); others 43.
case "$BATCH_TOKENS" in 20480) auto_slots=52 ;; 22528) auto_slots=50 ;; 24576) auto_slots=47 ;; *) auto_slots=43 ;; esac
LKQSTRATA_SLOTS=${LKQSTRATA_SLOTS:-$auto_slots}
# lkqmoe prefill pipeline: FP4 unpack and uploads on their own streams, the next layer's first N expert chunks
# uploaded during this layer's tail and the next attention layer. Bitwise identical; prefill +7-9% with N=6 at
# batch 22528 (+0.75 GiB at 522K). Only the validated batch gets it by default; 0 = off.
case "$BATCH_TOKENS" in 22528) auto_prefetch=6 ;; *) auto_prefetch=0 ;; esac
PREFILL_PREFETCH_CHUNKS=${PREFILL_PREFETCH_CHUNKS:-$auto_prefetch}
KV_TOKENS=${KV_TOKENS:-524288}         # context; pools above 786432 need LKQMOE_DS41_SPLIT_ZERO=1 (set below)
MAX_REQUESTS=${MAX_REQUESTS:-2}
GPU_MEMORY=${GPU_MEMORY:-0.95}
LKQ_THREADS=${LKQ_THREADS:-112}        # 64 = physical cores, 112 = + 6 SMT siblings per NUMA node (NPS4)
MAIN_CPUS=${MAIN_CPUS:-auto}           # host / loader CPUs: auto = every allowed CPU that is not an expert worker or the dispatcher

mkdir -p "$RUN_DIR/engram-stats" "$RUN_DIR/lkqmoe-stats" "$LOADER_SCRATCH"
touch "$RUN_DIR/enable-scale-ram"
overlay="$REPO/overlay"
export PYTHONPATH="$overlay${FLASHINFER_SM120:+:$FLASHINFER_SM120}:$REPO/adapter:$LSGLANG/python"
export DSV41_FROZEN_ROOT="$REPO"   # the overlay's sitecustomize runs $REPO/adapter/sitecustomize.py (Engram)

# ---- DSpark (block 5, 6 verify tokens, draft layers on GPU) ------------------------------
export DS41_ENABLE_FP8_VERIFY6_FASTPATH=1 SGLANG_TOOL_STRICT_LEVEL=0
unset SGLANG_DSPARK_STS_COLLECT_PATH SGLANG_DSPARK_FOLDED_PROPOSAL
export SGLANG_DSV41_REASONING_EFFORT=max
export LVLLM_GPU_RESIDENT_MOE_LAYERS_DSPARK=0-2
spec_args=(--speculative-draft-model-path "$DRAFT" --speculative-algorithm DSPARK
           --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6
           --speculative-dspark-sps-table-path "$overlay/sps-table.json")
check_sts(){
  "$VENV/bin/python" - "$STS_CALIBRATION" <<'CAL'
import json,math,sys
d=json.load(open(sys.argv[1])); t=d['temperatures']
assert len(t)==5 and d.get('num_samples',0)>0 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0 for x in t), 'Invalid STS calibration'
CAL
}
case "$VERIFY_MODE" in
  compact)
    # Variable-length verify was dropped on 2026-10-21: it changed output tokens (temperature-0 copy test, with one
    # and with two requests). Two requests decoded 13.9% faster, one request gained nothing.
    echo 'VERIFY_MODE=compact (variable-length verify) was dropped on 2026-10-21; use static or cap-sts' >&2; exit 2 ;;
  static) export SGLANG_RAGGED_VERIFY_MODE=static DS41_DSPARK_FULL_WIDTH=1 ;;
  cap-sts)
    check_sts
    export SGLANG_RAGGED_VERIFY_MODE=cap-accept DS41_DSPARK_FULL_WIDTH=1
    spec_args+=(--speculative-dspark-confidence-sts-path "$STS_CALIBRATION") ;;
  *) echo 'VERIFY_MODE must be static or cap-sts' >&2; exit 2 ;;
esac


# ---- Engram (NVMe row store), MoE placement, loader ---------------------------------------
export DSV41_HEALTH_URL="http://127.0.0.1:$PORT/health" DSV41_STATS_DIR="$RUN_DIR/engram-stats"
export DSV41_SCALE_RAM_TRIGGER="$RUN_DIR/enable-scale-ram"
export DSV41_SOURCE="$MODEL" DSV41_CACHE_GIB="$ENGRAM_CACHE_GIB" OFFLOAD_MODE=nvme
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0
if [[ "$MAIN_LAYERS" == 0 ]]; then resident_layers=""; else resident_layers="$RESIDENT_START-$((RESIDENT_START + MAIN_LAYERS - 1))"; fi
export LVLLM_MOE_NUMA_ENABLED=1 LVLLM_GPU_RESIDENT_MOE_LAYERS="$resident_layers"
export DS41_EARLY_LK_MOE=${DS41_EARLY_LK_MOE:-1}
# 2026-10-19: prefill indexer on DeepGEMM fp4 in row blocks, DSpark verify graph picked from a host
# length bound (no D2H sync), draft LM head in row-scaled FP8. 0 = the previous path.
export DS41_DENSE_FP4_INDEXER=${DS41_DENSE_FP4_INDEXER:-1} DS41_C128_HOST_BOUND=${DS41_C128_HOST_BOUND:-1} \
  DS41_DRAFT_LMHEAD_FP8=${DS41_DRAFT_LMHEAD_FP8:-1}
# Engram rows of layers 1 / 14 are looked up right after hashing on side streams (decode / verify);
# 0 = inline lookup. DS41_ENGRAM_IO_THREADS (default 16) = parallel O_DIRECT readers per lookup.
export DS41_ENGRAM_PREFETCH=${DS41_ENGRAM_PREFETCH:-1}
export SGLANG_LK_MOE_NVME_STAGING=1 SGLANG_LK_MOE_NVME_SCRATCH="$LOADER_SCRATCH"
export LVLLM_GPU_PREFETCH_WINDOW=1 LVLLM_ENABLE_NUMA_INTERLEAVE=1 LK_THREAD_BINDING=CPU_CORE
export LD_PRELOAD="$REPO/adapter/libnuma_fallback.so${LD_PRELOAD:+:$LD_PRELOAD}"
export MAX_JOBS=2 LK_THREADS=64 OMP_NUM_THREADS=1 LK_POWER_SAVING=1

# ---- CUDA / framework ----------------------------------------------------------------------
export CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export SGLANG_SKIP_P2P_CHECK=1 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLASHINFER_CUDA_ARCH_LIST=12.0 FLASHINFER_DISABLE_VERSION_CHECK=1
export TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=12.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ---- lkqmoe (compiled, ../lkqmoe) -----------------------------------------------------------
lkq="$REPO/lkqmoe"
export PYTHONPATH="$lkq/python:$PYTHONPATH"
export LKQMOE_MODE=standalone LKQMOE_CLOSED_FALLBACK=0 LKQMOE_CHAIN_SITECUSTOMIZE=1 \
  LKQMOE_LIBRARY="$lkq/liblkqmoe.so" LKQMOE_ZERO_COPY=1 LKQMOE_PREFILL_UNPACK=1 \
  LKQMOE_PREFILL_PREFETCH=1 LKQMOE_CPU_PREFILL_BATCH=2048 LKQMOE_SPIN_COUNT=1048576 \
  LKQMOE_ORIGINAL_WARMUP=0 LKQMOE_STATS_DIR="$RUN_DIR/lkqmoe-stats"
export LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=512     # lkqmoe GPU prefill from 512 tokens
export LKQMOE_RUNTIME_FILE="$RUN_DIR/lkqmoe-runtime.json"   # hybrid CPU+GPU prefill thresholds (re-read live)
[[ -f "$LKQMOE_RUNTIME_FILE" ]] || echo '{"hybrid_prefill_max": 6144, "cpu_prefill_below": 1024}' > "$LKQMOE_RUNTIME_FILE"
export LKQMOE_DOWN_PRECISION=tf32 LKQMOE_PREFILL_TILE_M=64 LKQMOE_GATE_BN=128 LKQMOE_GATE_LAUNCH=8,3 \
  LKQMOE_DOWN_BN=128 LKQMOE_DOWN_BK=64 LKQMOE_DOWN_LAUNCH=8,3   # GPU prefill tiles
export LKQMOE_THP=0                 # no MADV_HUGEPAGE on the expert shards (compaction stalls at load)
export LKQMOE_DOWN_BF16=1           # BF16 Down dot products on CPU (finer than the reference FP8 activations)
export LKQMOE_DYNAMIC=2             # guided dynamic row scheduling (bitwise identical to the static split)
export LKQMOE_DOWN_GRAIN=24         # 24-row dynamic Down claims, for the one-row Down build of lkqmoe 0.5.5 (bitwise identical)
export LKQMOE_FP8_CONFIGS=1         # tuned SM120 configs for the Triton block-FP8 GEMM
export LKQMOE_DS41_WO_A_TRITON=1    # decode wo_a on the Triton grouped einsum
export LKQMOE_DS41_WO_A_FP8=1       # keep wo_a in the checkpoint's FP8 (exact, 1.25 GiB less VRAM); 0 = BF16
export LKQMOE_PREFILL_EXPERT_CHUNK=8 LKQMOE_PREFILL_ROUTE_ACCUM=1   # GPU prefill: 8-expert chunks, FP32 route accumulation
# lkqmoe 0.5.6: GPU prefill experts on the FP4 tensor cores (FlashInfer 0.6.18 CUTLASS, MXFP4 x MXFP8 = the official activation
# precision), hot-pool experts copied on the GPU, 12 staging slots, <= 4096 tokens per call; prefill +6-10%. 0 = BF16 path.
export LKQMOE_PREFILL_FP4=${PREFILL_FP4:-1}
export LKQMOE_DS41_SPLIT_ZERO=1      # zero the reserved page of the SM120 page-split scratch (NaN fix, see README)
export LKQSTRATA="$LKQSTRATA" LKQSTRATA_SLOTS="$LKQSTRATA_SLOTS" LKQSTRATA_PROFILE="$lkq/lkqstrata-ds41-1017.json"
export LKQSTRATA_ADAPT=1 LKQSTRATA_LOG=500 LKQMOE_THREADS="$LKQ_THREADS"
export LKQSTRATA_EVERY=${LKQSTRATA_EVERY:-2} LKQSTRATA_SWAPS=${LKQSTRATA_SWAPS:-32}   # policy round / swap budget (2026-10-19; was 4 / 16)
if [[ "$PREFILL_PREFETCH_CHUNKS" -gt 0 ]]; then
  export LKQMOE_PREFILL_OVERLAP=1 LKQMOE_PREFILL_PREFETCH_CHUNKS="$PREFILL_PREFETCH_CHUNKS"
fi
# Mailbox dispatcher: CPU 95 with 112 workers (an SMT sibling no worker uses on 8 nodes x 14), CPU 80 with 64.
if [[ "$LKQ_THREADS" == 112 ]]; then
  export LKQMOE_THREADS=112 LKQMOE_PROBE_DISPATCH_CPU=95
else
  export LKQMOE_THREADS=64 LKQMOE_PROBE_DISPATCH_CPU=80
fi
# The worker CPUs come from lkqmoe_worker_cpus() of the library that will run (it reads the NUMA topology from sysfs).
# Host threads and the shard prefetch get the rest; an explicit MAIN_CPUS list is checked the same way. Startup stops
# if the result is empty, outside this process's CPU set, or overlaps a worker / the dispatcher. -S: no sitecustomize.
LKQMOE_MAIN_CPUS=$(python3 -S "$REPO/launch/resolve_main_cpus.py" --library "$LKQMOE_LIBRARY" \
  --threads "$LKQMOE_THREADS" --dispatcher "$LKQMOE_PROBE_DISPATCH_CPU" --main-cpus "$MAIN_CPUS")
export LKQMOE_MAIN_CPUS
echo "[ds41] host / loader CPUs: $LKQMOE_MAIN_CPUS" >&2

# ---- validate before allocating model weights ------------------------------------------
[[ -f "$MODEL/model.safetensors.index.json" && -f "$DRAFT/model.safetensors.index.json" ]] || { echo '1017 model or matching draft index missing' >&2; exit 3; }
[[ -d "$FLASHINFER_SM120/flashinfer" ]] || { echo 'Install the bundled FlashInfer 0.6.18 wheel into FLASHINFER_SM120 first' >&2; exit 4; }
"$VENV/bin/python" -c 'import sys,sitecustomize,lk_moe; assert sys.version_info[:2] == (3,12), "The compiled package requires Python 3.12"; assert getattr(lk_moe,"__lkqmoe_independent__",False) and hasattr(lk_moe,"MOE_MXFP4"), "Independent MXFP4 backend did not install"; assert any(type(x).__name__=="EngramFinder" for x in sys.meta_path), "Engram/early-load adapter did not install"'
(cd "$LSGLANG" && sha256sum --check --quiet "$REPO/patches/framework-files.sha256")
(cd "$REPO" && sha256sum --check --quiet SHA256SUMS)
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  printf 'CONFIG model=%s mode=%s main_layers=%s slots=%s kv=%s requests=%s early=%s batch=%s prefetch=%s threads=%s main_cpus=%s\n' "$MODEL" "$SGLANG_RAGGED_VERIFY_MODE" "$MAIN_LAYERS" "$LKQSTRATA_SLOTS" "$KV_TOKENS" "$MAX_REQUESTS" "$DS41_EARLY_LK_MOE" "$BATCH_TOKENS" "$PREFILL_PREFETCH_CHUNKS" "$LKQMOE_THREADS" "$LKQMOE_MAIN_CPUS"
  printf 'SPEC '; printf '%q ' "${spec_args[@]}"; printf '\n'
  exit 0
fi
scratch_free=$(df --output=avail -B1 "$LOADER_SCRATCH" | tail -n 1)
((scratch_free >= 320 * 1024 * 1024 * 1024)) || { echo 'Loader scratch needs at least 320 GiB free' >&2; exit 17; }

# ---- loading: sequential shard prefetch into page cache (random 4 KiB faults are slow) -----
python3 -S "$lkq/tools/shard_prefetch.py" "$MODEL" --lead 4 --threads 4 --cpus "$LKQMOE_MAIN_CPUS" \
  >> "$RUN_DIR/shard-prefetch.log" 2>&1 &

cd "$LSGLANG"
exec "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name DeepSeek-V4.1-Flash-1017 \
  --host "$HOST" --port "$PORT" --trust-remote-code --tensor-parallel-size 1 \
  --max-running-requests "$MAX_REQUESTS" --chunked-prefill-size "$BATCH_TOKENS" --max-prefill-tokens "$BATCH_TOKENS" \
  --context-length "$KV_TOKENS" --max-total-tokens "$KV_TOKENS" --mem-fraction-static "$GPU_MEMORY" \
  --moe-runner-backend flashinfer_cutlass --disable-shared-experts-fusion \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 "${spec_args[@]}" \
  --enable-decoder-swa-bounded-replay --enable-deepseek-v4-fp4-indexer \
  --weight-loader-drop-cache-after-load --model-loader-extra-config '{"enable_multithread_load": true}' \
  --watchdog-timeout 1200 --enable-metrics --enable-cache-report "$@"
