# lsglang_ds4.1

DeepSeek V4.1 Flash NVFP4 单卡推理管线：一张 RTX PRO 5000 72GB + 双路 EPYC，1M 上下文，DSpark 投机解码。
框架是 [guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)（sglang 的 CPU/GPU 混合推理分支），CPU 层的路由专家由受
[lk_moe](https://github.com/guqiong96/Lsglang)（lsglang 作者 guqiong96 的 CPU MoE 后端）启发、并针对 NVFP4 格式专项加速的
**lkqmoe** 承担计算（本仓库附编译好的闭源版本）。

*English: a single-GPU DeepSeek V4.1 Flash NVFP4 pipeline (RTX PRO 5000 72GB + 2x EPYC 9334, 1M context, DSpark speculative
decoding) on guqiong96/Lsglang, with the CPU-side routed experts on lkqmoe, an NVFP4-specialised MoE kernel inspired by lk_moe
(by guqiong96, the author of lsglang); a compiled, closed-source build is included here.*

## 组成

| 部分 | 来源 | 本仓库内容 |
|---|---|---|
| 目标模型 | [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4)（路由专家 NVFP4，其余线性层 FP8） | — |
| DSpark 草稿 | 官方 [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 检查点中的 `mtp.*` | `bench/make_draft_slim.py`（抽出 9.85 GiB 草稿专用副本） |
| 推理框架 | guqiong96/Lsglang @ [`6068667`](https://github.com/guqiong96/Lsglang/commit/6068667581bebd2184f7e4acb46136f44837889a) | `patches/lsglang-6068667.patch` |
| DSpark 运行时 | 在 Lsglang 基础上调试的稳定版模块（调度、验证、SPS 表等） | `overlay/runtime-dspark-stable-20260920-v2/` |
| Engram 行存储 | Engram 表放 NVMe，行缓存放内存 | `adapter/` |
| CPU 专家内核 | **lkqmoe**（闭源，编译版）：受 lk_moe 启发、针对 NVFP4 格式专项加速的 CPU/GPU 混合 MoE 内核。lk_moe 来自 lsglang 作者 guqiong96（[guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)）；lkqmoe 独立实现其 `MOE_NVFP4` 接口，不含 lk_moe 代码 | `lkqmoe/` |
| 启动 | 测速所用的完整参数 | `launch/run-ds41-nvfp4.sh` |

## 本机硬件

| | |
|---|---|
| CPU | 2× AMD EPYC 9334（共 64 核 128 线程），NPS2 → 4 个 NUMA 节点，AVX512-BF16/VBMI |
| 内存 | 320 GB DDR5-4800（20× 16 GB），实测带宽约 511 GB/s |
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell（SM120），驱动 595.84，CUDA 13.2 |
| 存储 | Engram 表与加载暂存用 Intel Optane P5800X，模型放普通 NVMe |
| 系统 | Ubuntu 24.04.4，内核 7.0.0-31，Python 3.12.3 |

## 速度

单请求，温度 0，`ignore_eos`，每点输出 1024 token；输入是长日志文本，在 10% 深度埋一条事实并在末尾提问
（`bench/speed_points.py --exact`）。KV 池 1,048,576，8 个点按 1M/8 等间隔取样。原始数据 `results/speed-points-1m-20261001.json`。

| 输入 token | 首 token 时间 | 预填充 tok/s | decode tok/s | 平均接受长度 | 找回埋藏事实 | 显存峰值 |
|---:|---:|---:|---:|---:|:---:|---:|
| 131,068 | 39.1 s | 3,355 | 55.2 | 3.66 | 是 | 72,362 MiB |
| 262,108 | 96.2 s | 2,726 | 58.1 | 3.64 | 是 | 72,364 MiB |
| 393,205 | 170.8 s | 2,302 | 52.7 | 3.57 | 是 | 72,740 MiB |
| 524,272 | 267.4 s | 1,961 | 55.7 | 3.54 | 是 | 72,620 MiB |
| 655,344 | 379.8 s | 1,726 | 52.3 | 3.50 | 是 | 72,824 MiB |
| 786,442 | 519.2 s | 1,515 | 49.5 | 3.46 | 是 | 72,820 MiB |
| 917,483 | 676.6 s | 1,356 | 47.7 | 3.40 | 是 | 72,800 MiB |
| 1,046,918 | 834.2 s | 1,255 | 50.1 | 3.39 | 是 | 72,822 MiB |

与官方原版（同一机器，官方检查点 + 闭源 lk_moe）的三点对比（`bench/ds41_3point.py`）：

| 输入 | 预填充 tok/s（本管线 / 官方） | decode tok/s（本管线 / 官方） |
|---:|---:|---:|
| 8,026 | 1,605 / 1,300 | 65.3 / 约 45 |
| 79,240 | 3,120 / 2,470 | 61.9 / 约 43 |
| 223,274 | 2,819 / 2,300 | 56.3 / 约 42 |

1M 满载时显存几乎用满，能跑通但余量小。

## 运行方式

- 目标模型 40 层：4 个 MoE 层常驻 GPU（第 20–23 层），其余 36 层的路由专家在 CPU（lkqmoe，112 线程）；3 个 DSpark 草稿层在 GPU。
- 共享专家、注意力、索引器在 GPU；KV 缓存 FP8，索引器 FP4；32K 分块预填充，SWA 尾部有界重放。
- DSpark：块大小 5，每步验证 6 个 token（`static` 全验证，贪心解码无损）。
- 预填充：短输入 CPU、长输入 GPU（lkqmoe 的 Triton 内核按 8 个专家一块上传权重，路由结果按块 FP32 累加），中间长度 CPU+GPU 混合。
- 注意力 `wo_a` 保持检查点原本的 FP8（从加载后的 BF16 精确还原、逐层核对），省 1.25 GiB 显存，误差不变。
- SM120 修复（`ds41_split_zero.py`）：SM120 稀疏注意力把无效 top-k 位置指向拆页暂存缓冲的保留页并乘 0 累加，而该页是未初始化显存，
  KV 池超过 768K 时会让每个提示前约 1K token 出现 NaN。这里在暂存缓冲分配时把保留页清零。

## 环境搭建

1. 框架：
   ```sh
   git clone https://github.com/guqiong96/Lsglang /opt/Lsglang && cd /opt/Lsglang
   git checkout 6068667581bebd2184f7e4acb46136f44837889a
   git apply /path/to/lsglang_ds4.1/patches/lsglang-6068667.patch
   ```
2. Python 3.12 虚拟环境（`/opt/Lsglang/env`），按 Lsglang v1.4.12 的 wheel 发布包安装，版本见 `requirements-lock.txt`
   （torch 2.13.0、triton 3.7.1、flashinfer 0.6.17、transformers 5.12.1）。不需要安装 `lk_moe`：lkqmoe 以 `LKQMOE_MODE=standalone`
   直接提供 `lk_moe` 模块。可选：SM120 版 FlashInfer 0.6.18，用 `FLASHINFER_SM120` 指向。
3. 模型：下载上面两个检查点；草稿副本 `python bench/make_draft_slim.py <官方检查点> <输出目录>`。
4. 按需修改 `launch/run-ds41-nvfp4.sh` 开头的路径后启动：`bash launch/run-ds41-nvfp4.sh`。
   服务在 `127.0.0.1:39503`，OpenAI 兼容接口，模型名 `DeepSeek-V4.1-Flash`，冷加载约 8–14 分钟。

需要约 290 GB 可用内存（CPU 专家分片约 270 GB + Engram 行缓存 12 GiB）。

## 主要启动参数

```
--chunked-prefill-size 32768 --max-prefill-tokens 32768 --context-length 1048576 --max-total-tokens 1048576
--mem-fraction-static 0.95 --max-running-requests 4 --moe-runner-backend flashinfer_cutlass --disable-shared-experts-fusion
--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
--speculative-algorithm DSPARK --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6
--enable-decoder-swa-bounded-replay --enable-deepseek-v4-fp4-indexer
--reasoning-parser deepseek-v41 --tool-call-parser deepseekv41
LVLLM_GPU_RESIDENT_MOE_LAYERS=20-23  LVLLM_GPU_RESIDENT_MOE_LAYERS_DSPARK=0-2  SGLANG_RAGGED_VERIFY_MODE=static
LKQMOE_THREADS=112  LKQMOE_DOWN_BF16=1  LKQMOE_DYNAMIC=2  LKQMOE_FP8_CONFIGS=1  LKQMOE_DS41_WO_A_TRITON=1
LKQMOE_DS41_WO_A_FP8=1  LKQMOE_PREFILL_EXPERT_CHUNK=8  LKQMOE_PREFILL_ROUTE_ACCUM=1  LKQMOE_DS41_SPLIT_ZERO=1
```
完整列表见 `launch/run-ds41-nvfp4.sh`。

## lkqmoe（`lkqmoe/`）

- 编译版：`liblkqmoe.so`（CPU 内核）、`liblkqmoe_cuda.so`（CUDA 信箱桥）、`python/lkqmoe/*.pyc`（lsglang 适配层与 Triton GPU 预填充；
  含 Triton 内核的模块内嵌压缩源码，因为 Triton 编译时要读源码）。许可见 `lkqmoe/LICENSE`：可免费使用、原样再分发。
- 开源部分（Apache-2.0）：`python/sitecustomize.py`；`python/lkqmoe/gpu/`：`wo_a` 的 Triton 分组 einsum 与 FP8 权重、SM120 拆页暂存修复、
  SM120 上 Triton 块 FP8 GEMM 的调优配置；`tools/shard_prefetch.py`（加载时顺序预读权重）。
- 精度：CPU 专家默认 BF16 Down（相对误差约 1.5e-3，优于官方参考实现的 FP8 激活）；GPU 预填充对 FP64 参考的误差 1.7e-3。
- 硬件要求：x86-64 AVX512-BF16（VBMI 更快），4 个 NUMA 节点 × 16 物理核（其他拓扑未验证），CUDA GPU。

## 测试脚本（`bench/`）

- `speed_points.py`：任意长度打点测速（本页速度表）；`--exact` 精确输入长度，`--stop-on-fail` 遇到失败点即停。
- `ds41_3point.py`、`ds41_longctx_test.py`：8.2K / 78.7K / 219.7K 三点测试与长上下文验收。
- `ds41_accept_bench.py`、`ds41_step_profile.py`、`ds41_interstep.py`：接受长度、单步耗时分解、步间空档分析。
- `ds41_fp8_tune.py`：SM120 上 Triton 块 FP8 GEMM 调优（生成 `lkqmoe/python/lkqmoe/gpu/fp8_configs/`）。

## 许可

本仓库开源部分按 Apache-2.0（与 Lsglang/sglang 相同）；`lkqmoe/` 下的二进制见其 LICENSE。模型权重遵循各自许可（MIT）。
