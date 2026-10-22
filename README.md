# DeepSeek V4.1 Flash 1017 · Lsglang + lkqmoe

官方 **DeepSeek-V4.1-Flash-1017（MXFP4）** 单卡推理框架：RTX PRO 5000 72GB、双路 EPYC，
**512K 共享 KV、最多 2 并发、DSpark 固定验证**，附编译好的 **lkqmoe 0.5.6**。
速度统一用编码代理测速 `bench/agent_bench.py`（多步工具调用，5 个会话从 64K 到约 490K，覆盖 512K KV 池）。

框架基于 [guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)；
lkqmoe 受该作者的 lk_moe CPU MoE 后端启发，是独立实现的通用 CPU/GPU MoE 内核，不包含 lk_moe 实现代码。
核心源码保留在私有仓库，本仓库提供编译包及开源框架适配。

English: DeepSeek V4.1 Flash 1017 MXFP4 on one GPU (RTX PRO 5000 72GB, dual EPYC) with 512K shared KV, two concurrent
requests, GPU-resident DSpark with a fixed 6-token verify and the compiled lkqmoe 0.5.6 backend. With 24 memory channels
and NPS4 use `launch/run-ds41-1017-24ch-nps4.sh` (agent_bench decode 157.5 / 144.0 tok/s in two rounds); with 20 channels
and NPS2 use `launch/run-ds41-1017.sh` (142.5 tok/s on 2026-10-20). What changed in each release is listed under
"版本更新", with the details in `reports/`.

## 版本更新

- **2026-10-22**：内存补满 24 通道（384 GiB）、BIOS 改 NPS4。新增启动脚本 `launch/run-ds41-1017-24ch-nps4.sh`，
  主线程 CPU 从 lkqmoe 的专家线程列表自动解析（原脚本的 NPS2 列表在 NPS4 下会与 8 个专家线程重叠）；默认参数不变，
  agent_bench decode 157.5 / 144.0 tok/s。详见 [reports/2026-10-22-24ch-nps4.md](reports/2026-10-22-24ch-nps4.md)（含测过但没有采用的调整）。
- **2026-10-21**：分词改为官方 deepseek-recipe 规则，内容里的控制 token 写法不再被编成真控制 token，长会话命中缓存的轮次首字 −18%；
  放弃 DSpark 变长验证（多请求时会改错 token），恢复固定验证 6 token。详见 [reports/2026-10-21-tokenizer-and-verify.md](reports/2026-10-21-tokenizer-and-verify.md)。
- **2026-10-20**：lkqmoe 0.5.6，GPU prefill 的专家走 FP4 张量核（MXFP4 × MXFP8），prefill +6–10%。
  详见 [reports/2026-10-20-fp4-prefill.md](reports/2026-10-20-fp4-prefill.md)。
- **2026-10-19**：prefill 索引器走 DeepGEMM FP4、DSpark 验证免同步、草稿词表头 FP8、Engram 提前查表与并行读、
  分块 22528 + 每层 50 个热专家槽、prefill 软件流水、lkqmoe 0.5.5（CPU Down 一次读一行）；测速统一为 agent_bench。
  详见 [reports/2026-10-19-optimizations.md](reports/2026-10-19-optimizations.md)。
- **2026-10-18**：以官方 1017 MXFP4 检查点和 lkqmoe 0.5.4 取代 NVFP4 预览版；OpenAI developer 角色兼容。
  详见 [reports/2026-10-18-release.md](reports/2026-10-18-release.md)。

## 组成

| 部分 | 当前版本 / 内容 |
|---|---|
| 模型与草稿 | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)，revision `1bae0e1088993b9b82de08f59de5be5a34e42170`；草稿从同一检查点提取 |
| Lsglang 基线 | `6068667581bebd2184f7e4acb46136f44837889a` |
| 1017 框架补丁 | `patches/lsglang-6068667-1017.patch`（14 个文件），含官方会话编码、block-32 激活量化、Engram 与 DSpark 调整，以及 10-19 的 decode 验证免同步和草稿词表头 FP8 |
| 运行时覆盖 | `overlay/`（含 10-19 的 prefill 索引器分块）；Engram 行存储（10-19 起提前查表、并行读）、量化和按层提前加载在 `adapter/` |
| lkqmoe | 0.5.6（10-20：GPU prefill 走 FP4 张量核 `LKQMOE_PREFILL_FP4`；动态库与 0.5.5 相同，AVX512 库用 `-DLKQMOE_DOWN_R4=1` 编译）；源码提交 `93edcc19eb06a71bccf8e8974d8c8c28decda344`；4 个动态库、Python 3.12 字节码、GPU 适配和热专家排名种子 |
| 启动入口 | `launch/run-ds41-1017.sh`（20 通道 / NPS2）；`launch/run-ds41-1017-24ch-nps4.sh`（24 通道 / NPS4，10-22） |
| 依赖 | `requirements-lock.txt`；附本次使用的 FlashInfer 0.6.18 和 b12x 1.3.0 轮子 |
| 实测与完整性 | `results/`、`SHA256SUMS` |

## 本机硬件与参数

| 项目 | 实测配置 |
|---|---|
| CPU | 2× EPYC 9334，64 核 / 128 线程 |
| 内存 / NUMA | 24 × 16 GiB DDR5-4800（384 GiB），NPS4、8 个 NUMA 节点（10-22 起；10-20 及以前为 20 通道 320 GB、NPS2、4 个节点） |
| GPU | RTX PRO 5000 72GB Blackwell（SM120） |
| 模型存储 | 三星 990 PRO；Engram 位于低延迟 Optane NVMe |
| 主模型专家 | 0 层整层常驻；40 层各使用 CPU + 50 个 LKQstrata GPU 热专家槽，自适应替换（每 2 个窗口一轮，每窗口最多换入 32 个） |
| DSpark | 3 层草稿常驻 GPU；块大小 5、每步固定验证 6 token |
| KV / 并发 | 524288 token，共享池最多 2 请求 |
| 预填充分块 / 显存比例 | 22528 / 0.95；prefill 软件流水，6 块跨层预取 |
| Engram / CPU | 行缓存 4 GiB；lkqmoe 112 线程（NPS4 下每节点 14 个）、BF16 Down、动态调度、Down 领取粒度 24 |
| 预填充切换阈值 | hybrid 上限 6144，CPU 阈值 1024 |
| 其他 | FP8 KV、FP4 indexer（prefill 也走 DeepGEMM）、SWA 尾部有界重放 |

两个请求共享 512K KV 池，不是每个请求各有 512K。
本机一次完整 static 启动约 **250–330 秒**；耗时会随页缓存和存储状态变化。

## 测速：编码代理（agent_bench）

`bench/agent_bench.py` 模拟编码代理会话（dsh / Claude Code 一类工具）。模型拿到一个真实代码任务，在沙箱副本里多步调用
read_file / list_dir / grep / edit_file / write_file / check_python，工具结果回送模型，后续请求命中前缀缓存。
沙箱内容是本仓库的 `adapter/`、`bench/` 和打过补丁的 Lsglang 源码，任务与文件完全公开，可以复现。

5 个会话的起点对齐原来的五点测速（约 64K / 128K / 256K / 384K / 480K），覆盖整个 512K KV 池：
- 每个会话先对整段 prompt 做一次冷 prefill，之后多步调用工具，上下文继续增长。
- 下一次请求可能超出 524288 时，会话提前结束。
- 每会话最多 24 步，采样用服务端默认（同 dsh，不指定温度）。
- 模型写出的代码只做语法检查，不会执行。

负载形态参照 10-19 一次真实 dsh 会话：77 个请求，前缀缓存命中 98.6%，大多是工具调用，夹杂长的写文件输出。

### 24 通道 / NPS4（2026-10-22，`launch/run-ds41-1017-24ch-nps4.sh`）

同一次启动连跑两轮，列与下面 20 通道的表相同。

第 1 轮：

| 会话 | 上下文（起 → 止） | 冷 prefill：首字 s / tok/s | 请求数 | 输出 token | decode tok/s | DSpark 接受长度 | 每步 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 64k | 66K → 90K | 19.6 / 3,377 | 17 | 4,943 | 146.5 | 4.65 | 31.6 |
| 128k-write | 132K → 168K | 31.6 / 4,187 | 14 | 4,967 | 150.9 | 4.75 | 31.4 |
| 256k | 265K → 346K | 64.2 / 4,122 | 24 | 1,316 | 195.2 | 5.33 | 26.8 |
| 384k | 394K → 517K | 99.2 / 3,973 | 15 | 811 | 207.7 | 5.55 | 26.2 |
| 500k | 489K → 519K | 126.9 / 3,856 | 11 | 592 | 211.2 | 5.53 | 25.7 |
| **全部** | | | 81 | 12,629 | **157.5** | 4.84 | 30.5 |

第 2 轮：

| 会话 | 上下文（起 → 止） | 冷 prefill：首字 s / tok/s | 请求数 | 输出 token | decode tok/s | DSpark 接受长度 | 每步 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 64k | 66K → 142K | 15.0 / 4,431 | 21 | 7,123 | 141.7 | 4.46 | 31.4 |
| 128k-write | 132K → 161K | 30.2 / 4,379 | 16 | 4,127 | 150.1 | 4.68 | 31.1 |
| 256k | 265K → 288K | 62.6 / 4,229 | 10 | 2,502 | 124.0 | 3.99 | 32.0 |
| 384k | 394K → 406K | 98.2 / 4,013 | 14 | 3,399 | 157.4 | 5.12 | 32.4 |
| 500k | 489K → 515K | 126.9 / 3,856 | 5 | 271 | 196.6 | 5.53 | 27.6 |
| **全部** | | | 66 | 17,422 | **144.0** | 4.56 | 31.6 |

- 两轮合计 30,051 个输出 token，decode 149.4 tok/s；命中前缀缓存的后续步骤首字中位 3.01 / 2.36 s。
- 两轮的输出量和接受长度差别较大（每轮内容由模型自己决定），所以给出两轮原始结果，不取单轮当结论。

[第 1 轮数据](results/agent-bench-20261022-24ch-nps4-r1.json) · [第 2 轮数据](results/agent-bench-20261022-24ch-nps4-r2.json)

### 20 通道 / NPS2（2026-10-20，`launch/run-ds41-1017.sh`）

| 会话 | 上下文（起 → 止） | 冷 prefill：首字 s / tok/s | 请求数 | 输出 token | decode tok/s | DSpark 接受长度 | 每步 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 64k | 66K → 106K | 18.9 / 3,516 | 24 | 6,186 | 143.0 | 4.74 | 33.0 |
| 128k-write | 132K → 157K | 32.0 / 4,132 | 24 | 3,766 | 136.3 | 4.43 | 32.3 |
| 256k | 265K → 401K | 63.9 / 4,141 | 24 | 1,191 | 195.2 | 5.37 | 26.9 |
| 384k | 394K → 417K | 99.1 / 3,978 | 18 | 4,606 | 147.2 | 5.03 | 34.0 |
| 500k | 489K → 518K | 127.3 / 3,844 | 19 | 5,856 | 135.3 | 4.68 | 34.5 |
| **全部** | | | 109 | 21,605 | **142.5** | 4.75 | 33.2 |

- 命中前缀缓存的后续步骤，首字中位 **2.05 s**。
- 本轮是 10-20 的生产配置（变长验证 + FP4 prefill）。变长验证 10-21 已放弃（见版本更新），固定验证下单请求 decode 约低 1–3%，两个请求同时 decode 合计约 100 tok/s。
- 输出长度 p10 / p50 / p90 为 47 / 77 / 479 token；≥ 500 token 的长输出（写文件、改代码）占输出 token 的 52%。

[原始数据](results/agent-bench-20261020.json)

### 计算口径

- 冷 prefill tok/s = 未命中缓存的 prompt token ÷ 首字时间。每轮测速的系统提示带有本轮编号，不会命中以前的缓存；同一轮的会话之间只共享约 1–2K token 的系统提示和工具定义。
- decode tok/s = 输出 token ÷ 首末流式片段间隔。
- 接受长度 = 输出 token ÷ DSpark 验证步数；每步 ms = decode 时间 ÷ 验证步数。
- 接受长度随内容和采样变化，同一配置两次运行约差 ±0.4。比较配置时先看"每步 ms"，decode tok/s 是这类任务上的实际体感。

## 运行方式

OpenAI消息入口支持 developer 角色：编码前按原顺序映射为模型原生 system，保留指令、文本块和工具字段；聊天、思考模式及已有上下文均兼容。此处理在历史思考内容裁剪之前完成，避免静默丢失应用指令。

目标专家保留紧凑的 MXFP4 权重和 UE8M0 block-32 scale，CPU 与 GPU 热专家池同时计算；
长预填充按专家分块上传 GPU。共享专家、注意力和索引器在 GPU。
默认 `VERIFY_MODE=static`（固定验证 6 个）。变长验证（`compact`）10-21 已放弃并从启动脚本移除。生产启动不采集 STS 样本。
`cap-sts` 仍可选，它加载 `config/dspark-sts-calib.json`；
校准缺失或无效会拒绝启动。当前完整验证预算固定为 1.0，不宣称 STS 带来性能提升。

## 环境搭建

需要 Linux x86-64、Python 3.12 和兼容的 CUDA 13 / SM120 环境。
按 [Lsglang 安装说明](https://github.com/guqiong96/Lsglang) 准备 1.4.12 运行环境；
本次 torch 2.13.0、Triton 3.7.1、transformers 5.12.1，参考版本见锁文件。
lkqmoe 独立提供兼容接口，不需要加载闭源 lk_moe。

```sh
git clone https://github.com/a775828b-dot/lsglang_ds4.1
export REPO="$(realpath lsglang_ds4.1)"
export LSGLANG="$HOME/Lsglang-1017"
git clone https://github.com/guqiong96/Lsglang "$LSGLANG"
git -C "$LSGLANG" checkout 6068667581bebd2184f7e4acb46136f44837889a
git -C "$LSGLANG" apply "$REPO/patches/lsglang-6068667-1017.patch"
export VENV="$LSGLANG/env"  # 已准备的 Python 3.12 / Lsglang 1.4.12 环境
"$VENV/bin/python" -m pip install --no-deps "$REPO/third_party/b12x-1.3.0-py3-none-any.whl"
export FLASHINFER_SM120="$VENV/flashinfer-0.6.18"
"$VENV/bin/python" -m pip install --no-deps --target "$FLASHINFER_SM120"   "$REPO/third_party/flashinfer_python-0.6.18-py3-none-any.whl"
```

下载上表固定 revision 的 1017 检查点，模型与草稿必须匹配。草稿约 9.85 GiB，提取不改变 tensor：

```sh
export MODEL=/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
export DRAFT="$MODEL-dspark"
"$VENV/bin/python" "$REPO/bench/make_draft_slim.py" "$MODEL" "$DRAFT"
export LOADER_SCRATCH=/scratch/ds41-1017
cd "$REPO"
sha256sum --check SHA256SUMS
# 24 通道 / NPS4：launch/run-ds41-1017-24ch-nps4.sh（主线程 CPU 自动解析）；20 通道 / NPS2：launch/run-ds41-1017.sh
CHECK_ONLY=1 bash launch/run-ds41-1017-24ch-nps4.sh
bash launch/run-ds41-1017-24ch-nps4.sh
```

默认接口 `http://127.0.0.1:39504/v1`，模型名 `DeepSeek-V4.1-Flash-1017`；
HOST、PORT 可配置。已有 8000 代理可以转发该后端，本仓库不包含本机代理密钥。
需要约 300 GiB 可用主机内存；启动器沿用至少 320 GiB scratch 空闲量检查。
Python 字节码固定为 3.12。当前整模型验收限于上述单卡硬件，不能把内核接口支持外推为 AVX2 主机或多卡 TP 的实测。

## 测速命令

```sh
"$VENV/bin/python" bench/agent_bench.py mytest --tokenizer "$MODEL" --lsglang "$LSGLANG" \
  --url http://127.0.0.1:39504 --out results/agent-bench-mytest.json
```

`--dry-run` 只构建沙箱和历史、不发请求；`--sessions`、`--steps`、`--temperature`、`--seed` 可调。需要 `tokenizers`（Lsglang 环境已带）。
性能测试避免并行其他推理负载或编译。当前 SWA bounded replay 不支持提示词 logprob / PPL 测试；
相关完整修复尚未移植。本文速度数据是文本推理及事实召回结果。

## lkqmoe 编译版

`lkqmoe/` 包含分发库、AVX512 库、AVX2 库、CUDA 桥与核心 Python 3.12 字节码。
Triton 模块沿用压缩源码封装并注册 linecache 的方式满足 JIT，不是不可逆保护。
GPU 适配与预读工具维持原有开源范围；原始核心 Python / C++ 文件不在此公开包中。

发布包已通过 CPU 数值兼容、AVX2、GPU packed/unpacked prefill、CUDA Graph 专家池、
300 窗口自适应换入换出、真实权重 FP64 参考，以及旧 wrapper / 新库 ABI 回归。
分发库与 AVX2 库的 AVX512 指令扫描均为 0。
[发布验证记录](results/kernel-release-validation.json) · [构建信息](lkqmoe/BUILD_INFO.json)。

整模型速度使用当前服务构建；发布包另行重编并做上述数值与 ABI 检查。
没有跨量化模型逐位输出一致或未测硬件性能的承诺。

## 许可

框架开源部分为 Apache-2.0；1017 会话编码的 MIT 许可见 third_party。
FlashInfer、b12x 轮子保留 Apache-2.0 许可。
lkqmoe 编译包采用 `lkqmoe/LICENSE`，核心源码在私有仓库；本仓库不含模型权重。
