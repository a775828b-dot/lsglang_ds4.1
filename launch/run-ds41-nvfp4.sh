#!/usr/bin/env bash
# DeepSeek V4.1 Flash NVFP4 + DSpark on one RTX PRO 5000 72GB (SM120) + 2x EPYC 9334 (4 NUMA nodes).
# Routed experts of the CPU layers run on lkqmoe (compiled, ../lkqmoe); 4 layers stay on the GPU.
# This is the launch used for the measurements in README.md; edit the paths block first.
set -Eeuo pipefail

# ---- paths ------------------------------------------------------------------------------
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
LSGLANG=${LSGLANG:-/opt/Lsglang}                       # guqiong96/Lsglang @ 6068667 + patches/lsglang-6068667.patch
VENV=${VENV:-$LSGLANG/env}                             # python 3.12, see README "Environment"
FLASHINFER_SM120=${FLASHINFER_SM120:-}                 # optional: FlashInfer 0.6.18 SM120 build (python path)
MODEL=${MODEL:-/models/nvidia/DeepSeek-V4.1-Flash-NVFP4}
DRAFT=${DRAFT:-/models/DeepSeek-V4.1-Flash-dspark-slim}  # bench/make_draft_slim.py, or the official checkpoint
LOADER_SCRATCH=${LOADER_SCRATCH:-/scratch/ds41-loader}   # NVMe, >= 320 GiB free (lk_moe NVMe staging while loading)
RUN_DIR=${RUN_DIR:-$REPO/run}
PORT=${PORT:-39503}

# ---- settings used for the published numbers --------------------------------------------
MAIN_LAYERS=${MAIN_LAYERS:-4}          # GPU-resident MoE layers of the target model (20-23)
RESIDENT_START=${RESIDENT_START:-20}
ENGRAM_CACHE_GIB=${ENGRAM_CACHE_GIB:-12}
BATCH_TOKENS=${BATCH_TOKENS:-32768}    # chunked prefill size
KV_TOKENS=${KV_TOKENS:-524288}
GPU_MEMORY=${GPU_MEMORY:-0.95}
LKQ_THREADS=${LKQ_THREADS:-112}        # 64 = physical cores, 112 = + 12 SMT siblings per NUMA node

mkdir -p "$RUN_DIR/engram-stats" "$RUN_DIR/lkqmoe-stats" "$LOADER_SCRATCH"
touch "$RUN_DIR/enable-scale-ram"
overlay="$REPO/overlay/runtime-dspark-stable-20260920-v2"
export PYTHONPATH="$overlay${FLASHINFER_SM120:+:$FLASHINFER_SM120}:$REPO/adapter:$LSGLANG/python"
export DSV41_FROZEN_ROOT="$REPO"   # the overlay's sitecustomize runs $REPO/adapter/sitecustomize.py (Engram)

# ---- DSpark (block 5, 6 verify tokens, draft layers on GPU) ------------------------------
export DS41_ENABLE_FP8_VERIFY6_FASTPATH=1 DS41_DSPARK_FULL_WIDTH=1 SGLANG_TOOL_STRICT_LEVEL=0
export SGLANG_RAGGED_VERIFY_MODE=static          # verify all draft tokens (lossless for greedy)
export LVLLM_GPU_RESIDENT_MOE_LAYERS_DSPARK=0-2
spec_args=(--speculative-draft-model-path "$DRAFT" --speculative-algorithm DSPARK
           --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6
           --speculative-dspark-sps-table-path "$overlay/sps-table.json")

# ---- Engram (NVMe row store), MoE placement, loader ---------------------------------------
export DSV41_HEALTH_URL="http://127.0.0.1:$PORT/health" DSV41_STATS_DIR="$RUN_DIR/engram-stats"
export DSV41_SCALE_RAM_TRIGGER="$RUN_DIR/enable-scale-ram"
export DSV41_SOURCE="$MODEL" DSV41_CACHE_GIB="$ENGRAM_CACHE_GIB" OFFLOAD_MODE=nvme
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0
export LVLLM_MOE_NUMA_ENABLED=1 LVLLM_GPU_RESIDENT_MOE_LAYERS="$RESIDENT_START-$((RESIDENT_START + MAIN_LAYERS - 1))"
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
export LKQMOE_MODE=standalone LKQMOE_CLOSED_FALLBACK=1 LKQMOE_CHAIN_SITECUSTOMIZE=1 \
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
export LKQMOE_FP8_CONFIGS=1         # tuned SM120 configs for the Triton block-FP8 GEMM
export LKQMOE_DS41_WO_A_TRITON=1    # decode wo_a on the Triton grouped einsum
if [[ "$LKQ_THREADS" == 112 ]]; then
  export LKQMOE_THREADS=112 LKQMOE_PROBE_DISPATCH_CPU=95 LKQMOE_MAIN_CPUS=76-79,92-94,108-111,124-127
else
  export LKQMOE_MAIN_CPUS=64-79,81-127
fi

# ---- loading: sequential shard prefetch into page cache (random 4 KiB faults are slow) -----
python3 -S "$lkq/tools/shard_prefetch.py" "$MODEL" --lead 4 --threads 4 --cpus "$LKQMOE_MAIN_CPUS" \
  >> "$RUN_DIR/shard-prefetch.log" 2>&1 &

cd "$LSGLANG"
exec "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name DeepSeek-V4.1-Flash \
  --host 127.0.0.1 --port "$PORT" --trust-remote-code --tensor-parallel-size 1 \
  --max-running-requests 4 --chunked-prefill-size "$BATCH_TOKENS" --max-prefill-tokens "$BATCH_TOKENS" \
  --context-length "$KV_TOKENS" --max-total-tokens "$KV_TOKENS" --mem-fraction-static "$GPU_MEMORY" \
  --moe-runner-backend flashinfer_cutlass --disable-shared-experts-fusion \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 "${spec_args[@]}" \
  --enable-decoder-swa-bounded-replay --enable-deepseek-v4-fp4-indexer \
  --weight-loader-drop-cache-after-load --model-loader-extra-config '{"enable_multithread_load": true}' \
  --watchdog-timeout 1200 --enable-metrics --enable-cache-report "$@"
