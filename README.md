# lsglang_ds4.1

DeepSeek V4.1 Flash NVFP4 单卡推理管线：一张 RTX PRO 5000 72GB + 双路 EPYC，512K 上下文、2 个请求并发，DSpark 投机解码，
**LKQstrata 热专家池**（CPU 层的热门专家另放 GPU、自适应换入换出，与 CPU 同时计算）。
框架是 [guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)（sglang 的 CPU/GPU 混合推理分支），CPU 层的路由专家由受
[lk_moe](https://github.com/guqiong96/Lsglang)（lsglang 作者 guqiong96 的 CPU MoE 后端）启发、并针对 NVFP4 格式专项加速的
**lkqmoe** 承担计算（本仓库附编译好的闭源版本）。

*English: a single-GPU DeepSeek V4.1 Flash NVFP4 pipeline (RTX PRO 5000 72GB + 2x EPYC 9334, 512K context, 2 concurrent requests, LKQstrata GPU expert pool, DSpark speculative
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
| 存储 | Engram 表与加载暂存用 Intel Optane P5800X，模型放三星 990 PRO（有 DRAM 缓存） |
| 系统 | Ubuntu 24.04.4，内核 7.0.0-31，Python 3.12.3 |

## 速度

**本版升级（2026-10-15，lkqmoe 0.5.0 LKQstrata 热专家池）**

- 上一版把第 20–23 层整层放在 GPU，其余 36 层的路由专家只在 CPU 上算，KV 池 1M。
- 本版只让解码段末尾的第 22–23 层整层常驻，其余 38 层每层 23 个"热专家"槽位（共 16.2 GiB）：decode 时槽位里的专家由 GPU 计算
  （W4A16），其余由 CPU 计算，两边同时进行后相加；槽位按排名表装填，运行中自适应换入换出。第 21–39 层预填充只算末尾 128 个
  token，所以首 token 时间不受影响。
- 专家池用掉了原先给 1M KV 的显存：上下文 1M → **512K**，同时运行的请求最多 2 个。
- 精度：GPU 承担的专家对 FP64 参考的误差 1.6e-3，与 CPU 路径同一量级；所有长度的埋藏事实都找回。
- 回到上一版（1M）：`LKQSTRATA=0 MAIN_LAYERS=4 RESIDENT_START=20 ENGRAM_CACHE_GIB=12 KV_TOKENS=1048576 MAX_REQUESTS=4 bash launch/run-ds41-nvfp4.sh`。

**单请求：上一版 → 本版**（温度 0，每点输出 1024 token，输入是埋了一条事实的长文本，`bench/speed_points.py --exact`；
上一版 2026-10-01，本版 2026-10-15；原始数据 `results/speed-points-1m-20261001.json`、`results/speed-points-20261015-lkqstrata.json`）

| 输入 token | decode tok/s | 首 token 时间 |
|---:|---:|---:|
| 131K | 55.2 → **73.7**（+34%） | 39.1 → 38.4 s |
| 262K | 58.1 → **92.9**（+60%） | 96.2 → 93.7 s |
| 393K | 52.7 → **63.7**（+21%） | 170.8 → 168.0 s |
| 524K → 500K | 55.7 → **87.9**（+58%） | 267.4 → 245.0 s |
| **平均** | 55.4 → **79.6（+44%）** | |

**并发：上一版 → 本版**（短提示，温度 1.0，每个请求 512 token，同时开始，各 4 轮，`bench/conc_chat.py`，同一晚测得，
两版都按 512K、最多 2 个请求运行；`results/ab-20261015-lkqstrata.json`）

| 并发 | 上一版 单请求 / 总吞吐 | 本版 单请求 / 总吞吐 | 总吞吐 |
|---:|---:|---:|---:|
| 1 | 63.6 / 63.8 tok/s | 82.8 / 82.9 tok/s | **+30%** |
| 2 | 32.6 / 59.3 tok/s | 41.7 / 76.5 tok/s | **+29%** |

两个请求共用 524,288 token 的 KV 池，单个对话可以用满 512K。

## 运行方式

- 目标模型 40 层：本版第 22–23 层整层常驻 GPU，其余 38 层的路由专家在 CPU（lkqmoe，112 线程），每层另有 23 个热专家槽位在 GPU
  （LKQstrata）；上一版是第 20–23 层常驻、36 层在 CPU。3 个 DSpark 草稿层在 GPU。
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
   服务在 `127.0.0.1:39503`，OpenAI 兼容接口，模型名 `DeepSeek-V4.1-Flash`。加载器按 4 KiB 页随机读多个分片，模型盘要用有 DRAM
   缓存的 NVMe：本机三星 990 PRO 上约 4 分钟读完权重、6 分钟可用；无 DRAM（HMB）的盘在这种读法下会掉到几十 MB/s，要半小时以上。

需要约 300 GB 可用内存（本版 38 层 CPU 专家分片约 285 GB + Engram 行缓存 4 GiB；上一版约 290 GB）。

## 主要启动参数

```
--chunked-prefill-size 32768 --max-prefill-tokens 32768 --context-length 524288 --max-total-tokens 524288
--mem-fraction-static 0.95 --max-running-requests 2 --moe-runner-backend flashinfer_cutlass --disable-shared-experts-fusion
--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
--speculative-algorithm DSPARK --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6
--enable-decoder-swa-bounded-replay --enable-deepseek-v4-fp4-indexer
--reasoning-parser deepseek-v41 --tool-call-parser deepseekv41
LVLLM_GPU_RESIDENT_MOE_LAYERS=22-23  LVLLM_GPU_RESIDENT_MOE_LAYERS_DSPARK=0-2  DSV41_CACHE_GIB=4
LKQSTRATA=1  LKQSTRATA_SLOTS=23  LKQSTRATA_PROFILE=lkqmoe/lkqstrata-ds41.json  SGLANG_RAGGED_VERIFY_MODE=static
LKQMOE_THREADS=112  LKQMOE_DOWN_BF16=1  LKQMOE_DYNAMIC=2  LKQMOE_FP8_CONFIGS=1  LKQMOE_DS41_WO_A_TRITON=1
LKQMOE_DS41_WO_A_FP8=1  LKQMOE_PREFILL_EXPERT_CHUNK=8  LKQMOE_PREFILL_ROUTE_ACCUM=1  LKQMOE_DS41_SPLIT_ZERO=1
```
完整列表见 `launch/run-ds41-nvfp4.sh`。

## lkqmoe（`lkqmoe/`）

- 编译版：`liblkqmoe.so`（CPU 内核）、`liblkqmoe_cuda.so`（CUDA 信箱桥）、`python/lkqmoe/*.pyc`（lsglang 适配层与 Triton GPU 预填充；
  含 Triton 内核的模块内嵌压缩源码，因为 Triton 编译时要读源码）。许可见 `lkqmoe/LICENSE`：可免费使用、原样再分发。
- 开源部分（Apache-2.0）：`python/sitecustomize.py`；`python/lkqmoe/gpu/`：`wo_a` 的 Triton 分组 einsum 与 FP8 权重、SM120 拆页暂存修复、
  SM120 上 Triton 块 FP8 GEMM 的调优配置；`tools/shard_prefetch.py`（加载时顺序预读权重）。
- 0.5.0（2026-10-15）：LKQstrata 热专家池（`pool.pyc`，排名表 `lkqstrata-ds41.json`）——CPU 层的热门专家另放 GPU 槽位，decode 时 GPU
  与 CPU 并行计算后相加；`LKQSTRATA=0` 时与 0.4.x 的数值路径相同。与 lsglang_glm5.3、lsglang_q3.8fn 是同一份二进制与源码。
- 精度：CPU 专家默认 BF16 Down（相对误差约 1.5e-3，优于官方参考实现的 FP8 激活）；GPU 预填充对 FP64 参考的误差 1.7e-3。
- 硬件要求：x86-64 AVX512-BF16（VBMI 更快），4 个 NUMA 节点 × 16 物理核（其他拓扑未验证），CUDA GPU。

## 测试脚本（`bench/`）

- `speed_points.py`：任意长度打点测速（本页速度表）；`--exact` 精确输入长度，`--stop-on-fail` 遇到失败点即停。
- `ds41_3point.py`、`ds41_longctx_test.py`：8.2K / 78.7K / 219.7K 三点测试与长上下文验收。
- `ds41_accept_bench.py`、`ds41_step_profile.py`、`ds41_interstep.py`：接受长度、单步耗时分解、步间空档分析。
- `step_chat.py`：贪心解码下按流式块间隔测每步耗时（1 / 2 并发，不受接受长度影响）；`conc_chat.py`：温度 1.0 并发吞吐。
- `ds41_fp8_tune.py`：SM120 上 Triton 块 FP8 GEMM 调优（生成 `lkqmoe/python/lkqmoe/gpu/fp8_configs/`）。

## 许可

本仓库开源部分按 Apache-2.0（与 Lsglang/sglang 相同）；`lkqmoe/` 下的二进制见其 LICENSE。模型权重遵循各自许可（MIT）。
