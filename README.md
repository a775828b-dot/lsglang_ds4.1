# lsglang_ds4.1

DeepSeek V4.1 Flash NVFP4 单卡推理管线：一张 RTX PRO 5000 72GB + 双路 EPYC，768K 上下文（实测上限），DSpark 投机解码。
框架是 [guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)（sglang 的 CPU/GPU 混合推理分支），CPU 层的路由专家由受
[lk_moe](https://github.com/guqiong96/Lsglang)（lsglang 作者 guqiong96 的 CPU MoE 后端）启发、并针对 NVFP4 格式专项加速的
**lkqmoe** 承担计算（本仓库附编译好的闭源版本）。

*English: a single-GPU DeepSeek V4.1 Flash NVFP4 pipeline (RTX PRO 5000 72GB + 2x EPYC 9334, 768K context (measured limit), DSpark speculative
decoding) on guqiong96/Lsglang, with the CPU-side routed experts on lkqmoe, an NVFP4-specialised MoE kernel inspired by lk_moe (by guqiong96, the author of lsglang); a compiled, closed-source build is included here.
Speeds at 8 context lengths from 4K to 500K and a context ladder up to 896K are below.*

## 组成

| 部分 | 来源 | 本仓库内容 |
|---|---|---|
| 目标模型 | [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4)（路由专家 NVFP4，其余线性层 FP8） | — |
| DSpark 草稿 | 官方 [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 检查点中的 `mtp.*` | `bench/make_draft_slim.py`（抽出 9.85 GiB 草稿专用副本，加快加载） |
| 推理框架 | guqiong96/Lsglang @ [`6068667`](https://github.com/guqiong96/Lsglang/commit/6068667581bebd2184f7e4acb46136f44837889a) | `patches/lsglang-6068667.patch`（7 个文件） |
| DSpark 运行时 | 在 Lsglang 基础上调试的稳定版模块（调度、验证、SPS 表等） | `overlay/runtime-dspark-stable-20260920-v2/` |
| Engram 行存储 | Engram 表放 NVMe，行缓存放内存 | `adapter/`（源码 + 编译好的 .so） |
| CPU 专家内核 | **lkqmoe**（闭源，编译版）：受 lk_moe 启发、针对 NVFP4 格式专项加速的 CPU/GPU 混合 MoE 内核。lk_moe 来自 lsglang 作者 guqiong96（[guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)）；lkqmoe 独立实现其 `MOE_NVFP4` 接口，不含 lk_moe 代码 | `lkqmoe/` |
| 启动 | 本文测速所用的完整参数 | `launch/run-ds41-nvfp4.sh` |

## 本机硬件

| | |
|---|---|
| CPU | 2× AMD EPYC 9334（32 核/颗，共 64 核 128 线程），NPS2 → 4 个 NUMA 节点，256 MB L3，AVX512-BF16/VBMI |
| 内存 | 320 GB DDR5（20× 16 GB，4800 MT/s），实测带宽约 511 GB/s |
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell（SM120），驱动 595.84，CUDA 13.2 |
| 存储 | NVMe：Engram 表与加载暂存用 Intel Optane P5800X；模型放普通 NVMe |
| 系统 | Ubuntu 24.04.4，内核 7.0.0-31，Python 3.12.3 |

## 速度

测试条件：单请求，温度 0，`ignore_eos`，每点输出 1024 token；输入是长日志文本，在 10% 深度埋一条事实并在末尾提问（`bench/speed_points.py`）。DSpark 开启，4 层 GPU 常驻，其余见“运行方式”。2026-09-30 实测，原始数据 `results/speed-points-20260930.json`。

| 输入 token | 首 token 时间 | 预填充 tok/s | decode tok/s | 平均接受长度 | 找回埋藏事实 | 显存峰值 |
|---:|---:|---:|---:|---:|:---:|---:|
| 4,012 | 4.4 s | 909¹ | 56.6 | 3.23 | 是 | 70,822 MiB |
| 15,882 | 4.9 s | 3,230 | 58.0 | 3.26 | 是 | 70,822 MiB |
| 32,660 | 8.2 s | 3,968 | 51.5 | 3.22 | 是 | 70,824 MiB |
| 65,853 | 18.4 s | 3,575 | 58.5 | 3.27 | 是 | 70,824 MiB |
| 129,901 | 39.0 s | 3,330 | 45.5 | 3.19 | 是 | 70,828 MiB |
| 272,721 | 104.5 s | 2,611 | 52.3 | 3.17 | 是 | 71,470 MiB |
| 409,503 | 184.5 s | 2,219 | 52.8 | 3.18 | 是 | 71,966 MiB |
| 521,055 | 267.9 s | 1,945 | 55.4 | 3.18 | 是 | 72,806 MiB |

¹ 服务启动后的第一个请求，包含 Triton 内核的一次性编译；之后同尺寸请求不再有这部分开销。

500K 档（实际 521K 输入 + 1K 输出）即 512K 上下文的容量验收：显存峰值 72,806 MiB（总 73,415 MiB）。

### 上下文上限（2026-10-01）

注意力的 `wo_a` 改为保持检查点原本的 FP8 后（见“运行方式”），省下 1.25 GiB 显存，用来扩大 KV 池。测法：KV 池设为 1M，
输入从 500K（512,000 token）起每档加 128K，每档输出 2K，其余同上；原始数据 `results/kv-ladder-20261001.json`。

| 输入 token | 首 token 时间 | 预填充 tok/s | decode tok/s | 接受长度 | 找回埋藏事实 |
|---:|---:|---:|---:|---:|:---:|
| 511,961 | 262.8 s | 1,948 | 68.0 | 3.77 | 是 |
| 643,054 | 374.0 s | 1,719 | 59.4 | 3.74 | 是 |
| 774,103 | 509.8 s | 1,519 | 60.7 | 3.74 | 是 |
| 905,216 | — | — | — | — | 预填充到约 754K 时显存不足（OOM），服务退出 |

预填充时注意力与索引器的工作区随上下文增长（约 5.4 MiB / 1K token），1M 的 KV 池本身也比 512K 多占约 0.9 GB，所以这张卡上
上限是 768K。KV 池设为 786,432 后按满容量复测：782,979 输入 + 2,048 输出通过（首 token 523 s，decode 62.8 tok/s，
`results/kv-768k-check-20261001.json`）。这一档显存几乎用满（分配器需要回收缓存才能继续），余量很小；要更稳可用 640K
（655,360）。

与官方原版（同一机器，官方 DeepSeek V4.1 检查点 + 闭源 lk_moe，2026-09-17）对比，三点测试 `bench/ds41_3point.py`：

| 输入 | 预填充 tok/s（本管线 / 官方） | decode tok/s（本管线 / 官方） |
|---:|---:|---:|
| 8,026 | 1,198¹ / 1,300 | 60.6 / 约 45 |
| 79,240 | 2,924 / 2,470 | 50.8 / 约 43 |
| 223,274 | 2,798 / 2,300 | 46.9 / 约 42 |

官方原版 decode 从约 45 逐步降到约 41 tok/s。

2026-10-01 开启 FP8 `wo_a`、KV 池 1M 后复测三点：预填充 1,605 / 3,120 / 2,816 tok/s，decode 65.6 / 51.7 / 70.7 tok/s
（`results/3point-woafp8-20261001.json`）。

## 运行方式

- 4 个 MoE 层常驻 GPU（第 20–23 层），3 个 DSpark 草稿层在 GPU；目标模型共 40 层，其余 36 层的路由专家在 CPU（lkqmoe，112 线程）。
- 共享专家、注意力、索引器全在 GPU；KV 缓存 FP8（768K 约 3.0 GB，其中固定的 SWA 部分 1.7 GB），索引器 FP4。
- 注意力的 `wo_a` 保持检查点原本的 FP8（E4M3 + 32×32 的 2 的幂次缩放）：框架加载时展开成 BF16，这里精确还原回 FP8，逐层核对后
  释放 BF16 副本，省 1.25 GiB。decode 用 FP8 版 Triton 分组 einsum（与 BF16 版对 FP64 参考的误差相同），预填充反量化到复用缓冲后
  沿用原 einsum（逐位一致）。`LKQMOE_DS41_WO_A_FP8=0` 恢复 BF16。
- DSpark：块大小 5，每步验证 6 个 token（`static` 全验证，贪心解码无损）。
- 预填充：短输入 CPU、长输入 GPU（lkqmoe 按专家分块上传权重的 Triton 内核），中间长度 CPU+GPU 混合；32K 分块。
- SWA 尾部有界重放（`--enable-decoder-swa-bounded-replay`），FP4 索引器（`--enable-deepseek-v4-fp4-indexer`）。

## 环境搭建

1. 框架：
   ```sh
   git clone https://github.com/guqiong96/Lsglang /opt/Lsglang && cd /opt/Lsglang
   git checkout 6068667581bebd2184f7e4acb46136f44837889a
   git apply /path/to/lsglang_ds4.1/patches/lsglang-6068667.patch
   ```
2. Python 3.12 虚拟环境（`/opt/Lsglang/env`），按 Lsglang v1.4.12 的 wheel 发布包安装，版本见 `requirements-lock.txt`
   （torch 2.13.0、triton 3.7.1、flashinfer 0.6.17、transformers 5.12.1）。
   **不需要安装 `lk_moe`**：`LKQMOE_MODE=standalone`（`LKQMOE_CLOSED_FALLBACK=0`）时 lkqmoe 直接提供 `lk_moe` 模块，闭源包不会被加载。
   这套配置里只有 NVFP4 的 CPU 层用到该接口；DSpark 草稿层（MXFP4）全部常驻 GPU，不经过它。可选：SM120 版 FlashInfer 0.6.18，用 `FLASHINFER_SM120` 指向。
3. 模型：下载上面两个检查点；草稿副本 `python bench/make_draft_slim.py <官方检查点> <输出目录>`。
4. 按需修改 `launch/run-ds41-nvfp4.sh` 开头的路径，启动：
   ```sh
   bash launch/run-ds41-nvfp4.sh
   ```
   服务在 `127.0.0.1:39503`，OpenAI 兼容接口，模型名 `DeepSeek-V4.1-Flash`。加载约 8–13 分钟（冷页缓存）。

内存：常驻约 270 GB（CPU 专家分片 + Engram 行缓存 12 GiB），需要约 290 GB 可用内存再启动。

## 主要启动参数

```
--chunked-prefill-size 32768 --max-prefill-tokens 32768 --context-length 786432 --max-total-tokens 786432
--mem-fraction-static 0.95 --max-running-requests 4 --moe-runner-backend flashinfer_cutlass --disable-shared-experts-fusion
--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
--speculative-algorithm DSPARK --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6
--enable-decoder-swa-bounded-replay --enable-deepseek-v4-fp4-indexer
--reasoning-parser deepseek-v41 --tool-call-parser deepseekv41
LVLLM_GPU_RESIDENT_MOE_LAYERS=20-23  LVLLM_GPU_RESIDENT_MOE_LAYERS_DSPARK=0-2  SGLANG_RAGGED_VERIFY_MODE=static
LKQMOE_THREADS=112  LKQMOE_DOWN_BF16=1  LKQMOE_DYNAMIC=2  LKQMOE_FP8_CONFIGS=1  LKQMOE_DS41_WO_A_TRITON=1
LKQMOE_DS41_WO_A_FP8=1
```
完整列表见 `launch/run-ds41-nvfp4.sh`。上面 8 点速度表是 KV 池 524,288 时测的。

## lkqmoe（`lkqmoe/`）

- `liblkqmoe.so`（CPU 内核）、`liblkqmoe_cuda.so`（CUDA 信箱桥，可被 CUDA Graph 捕获）、`python/lkqmoe/*.pyc`（lsglang 适配层与
  Triton GPU 预填充，编译版；含 Triton 内核的模块把源码压缩内嵌，因为 Triton 编译时需要读取源码）。
- 开源部分（Apache-2.0）：`python/sitecustomize.py`、`python/lkqmoe/gpu/`（DeepSeek `wo_a` 的 Triton 分组 einsum 与 FP8 权重、
  SM120 下 Triton 块 FP8 GEMM 的调优配置）、`tools/shard_prefetch.py`（加载时顺序预读权重）。
- 二进制许可见 `lkqmoe/LICENSE`：可免费使用、原样再分发；源码不公开。
- 硬件要求：x86-64 AVX512-BF16（VBMI 更快），4 个 NUMA 节点 × 16 物理核（其他拓扑未验证），CUDA GPU。

与闭源 `lk_moe` 相比：CPU 专家 FP32 模式误差 2.2e-7（lk_moe 8e-7），DS4.1 默认 BF16 Down（相对误差约 1.5e-3，仍优于官方参考实现的
FP8 激活）；GPU 预填充每层 8192 token 142 ms（lk_moe 600 ms）；decode 验证步详见速度表。

## 测试脚本（`bench/`）

- `speed_points.py`：任意长度打点测速（本页速度表）；`--exact` 把输入长度校准到目标 ±0.05%，`--stop-on-fail` 遇到第一个失败点即停
  （上下文上限阶梯）。
- `ds41_3point.py`、`ds41_longctx_test.py`：8.2K / 78.7K / 219.7K 三点对比与 500K 长上下文验收。
- `ds41_accept_bench.py`、`ds41_step_profile.py`、`ds41_interstep.py`：接受长度、单步耗时分解、步间空档分析。
- `ds41_fp8_tune.py`：SM120 上 Triton 块 FP8 GEMM 的调优（生成 `lkqmoe/python/lkqmoe/gpu/fp8_configs/`）。

## 许可

本仓库开源部分按 Apache-2.0（与 Lsglang/sglang 相同）；`lkqmoe/` 下的二进制见其 LICENSE。模型权重遵循各自许可（MIT）。
