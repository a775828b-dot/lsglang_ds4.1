# DeepSeek V4.1 Flash 1017 · Lsglang + lkqmoe

官方 **DeepSeek-V4.1-Flash-1017（MXFP4）** 单卡推理框架：RTX PRO 5000 72GB、双路 EPYC，
**512K 共享 KV、最多 2 并发、DSpark、默认 static 验证**，附编译好的 **lkqmoe 0.5.4**。
本页参数和速度来自 2026-10-19 午的部署：prefill 分块 22528、每层 50 个热专家槽、prefill 软件流水（6 块跨层预取）。
速度统一用编码代理测速 `bench/agent_bench.py`（多步工具调用，上下文约 20K–180K），见下文。

框架基于 [guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)；
lkqmoe 受该作者的 lk_moe CPU MoE 后端启发，是独立实现的通用 CPU/GPU MoE 内核，不包含 lk_moe 实现代码。
核心源码保留在私有仓库，本仓库提供编译包及开源框架适配。

English: DeepSeek V4.1 Flash 1017 MXFP4 on one GPU, with 512K shared KV, two concurrent
requests, GPU-resident DSpark and the compiled lkqmoe 0.5.4 backend. The 2026-10-19 update keeps
long prefills on the DeepGEMM FP4 indexer and takes the Engram row lookups off the decode critical path; the midday
update uses a 22528-token prefill chunk, 50 hot-expert GPU slots per layer and lkqmoe's prefill pipeline. Speeds come
from the coding-agent benchmark `bench/agent_bench.py` (multi-step tool use, 20K-180K context).

## 组成

| 部分 | 当前版本 / 内容 |
|---|---|
| 模型与草稿 | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)，revision `1bae0e1088993b9b82de08f59de5be5a34e42170`；草稿从同一检查点提取 |
| Lsglang 基线 | `6068667581bebd2184f7e4acb46136f44837889a` |
| 1017 框架补丁 | `patches/lsglang-6068667-1017.patch`（14 个文件），含官方会话编码、block-32 激活量化、Engram 与 DSpark 调整，以及 10-19 的 decode 验证免同步和草稿词表头 FP8 |
| 运行时覆盖 | `overlay/`（含 10-19 的 prefill 索引器分块）；Engram 行存储（10-19 起提前查表、并行读）、量化和按层提前加载在 `adapter/` |
| lkqmoe | 0.5.4（10-19 的更新均未改动内核与编译库）；源码提交 `9cbeb6c7a8a3bd2fc36ced525004450eb3a1dee5`；4 个动态库、Python 3.12 字节码、GPU 适配和热专家排名种子 |
| 启动入口 | `launch/run-ds41-1017.sh` |
| 依赖 | `requirements-lock.txt`；附本次使用的 FlashInfer 0.6.18 和 b12x 1.3.0 轮子 |
| 实测与完整性 | `results/`、`SHA256SUMS` |

## 本机硬件与参数

| 项目 | 实测配置 |
|---|---|
| CPU | 2× EPYC 9334，64 核 / 128 线程，NPS2、4 个 NUMA 节点 |
| 内存 / GPU | 320 GB DDR5；RTX PRO 5000 72GB Blackwell（SM120） |
| 模型存储 | 三星 990 PRO；Engram 位于低延迟 Optane NVMe |
| 主模型专家 | 0 层整层常驻；40 层各使用 CPU + 50 个 LKQstrata GPU 热专家槽（分块 32768 时为 43），自适应替换（每 2 个窗口一轮，每窗口最多换入 32 个） |
| DSpark | 3 层草稿常驻 GPU；块大小 5、每步验证 6 token；static |
| KV / 并发 | 524288 token，共享池最多 2 请求 |
| 预填充分块 / 显存比例 | 22528 / 0.95；prefill 软件流水，6 块跨层预取 |
| Engram / CPU | 行缓存 4 GiB；112 线程、BF16 Down、动态调度 |
| 预填充切换阈值 | hybrid 上限 6144，CPU 阈值 1024 |
| 其他 | FP8 KV、FP4 indexer（prefill 也走 DeepGEMM）、SWA 尾部有界重放 |

两个请求共享 512K KV 池，不是每个请求各有 512K。
每层 CPU 专家加载完成后立即交给 lkqmoe 并释放临时 scratch；`DS41_EARLY_LK_MOE=1` 默认开启。
本机一次完整 static 启动约 **310–330 秒**；权重加载记录约 **200 秒**，耗时会随页缓存和存储状态变化。

## 测速：编码代理（agent_bench）

`bench/agent_bench.py` 模拟编码代理会话（dsh / Claude Code 一类工具）。模型拿到一个真实代码任务，在沙箱副本里多步调用
read_file / list_dir / grep / edit_file / write_file / check_python，工具结果回送模型，后续请求命中前缀缓存。
沙箱内容是本仓库的 `adapter/`、`bench/` 和打过补丁的 Lsglang 源码，任务与文件完全公开，可以复现。
4 个会话的预填上下文约 21K / 52K / 86K / 161K，对照 10-19 一次真实 dsh 会话：77 个请求，上下文 p10/50/90 为
14K / 84K / 168K，前缀缓存命中 98.6%。每会话最多 24 步，采样用服务端默认（同 dsh，不指定温度）。
模型写出的代码只做语法检查，不会执行。

**decode（按上下文分档）**

| 上下文 | 请求数 | 输出 token | decode tok/s | 单请求中位 tok/s | DSpark 接受长度 | 每步 ms |
|---|---:|---:|---:|---:|---:|---:|
| ≤ 32K | 3 | 162 | 170.4 | 186.4 | 5.40 | 31.1 |
| 32K–64K | 19 | 11,763 | 134.6 | 143.4 | 4.73 | 35.1 |
| 64K–128K | 21 | 4,197 | 125.4 | 176.7 | 4.17 | 33.1 |
| ≥ 128K | 9 | 3,114 | 141.5 | 161.7 | 4.94 | 34.8 |
| **全部** | 52 | 19,236 | **133.8** | 162.5 | 4.63 | 34.5 |

**冷 prefill（每个会话第一步，整段 prompt 不命中缓存）**

| 会话 | prompt token | 首字 s | prefill tok/s |
|---|---:|---:|---:|
| 20k | 21,134 | 5.4 | 3,928 |
| 50k-write | 52,291 | 14.1 | 3,716 |
| 85k | 86,922 | 22.0 | 3,947 |
| 160k | 161,883 | 42.1 | 3,847 |

- 后续步骤（命中前缀缓存）首字中位 **1.77 s**。
- 输出长度 p10 / p50 / p90 为 46 / 80 / 870 token；≥ 500 token 的长输出（写文件、改代码）占输出 token 的 78%。
- decode tok/s = 输出 token ÷ 首末流式片段间隔；接受长度 = 输出 token ÷ DSpark 验证步数；每步 ms = decode 时间 ÷ 验证步数。
- 接受长度随内容和采样变化，同一配置两次运行约差 ±0.4。比较配置时先看"每步 ms"，decode tok/s 是这类任务上的实际体感。
- 旧的五点、多领域 decode 和并发测速已从仓库移除，统一用本测速。

[原始数据](results/agent-bench-20261019.json)

**满容量验证（不是测速）**：522,300 输入 + 1,024 输出（KV 池 524288）完成且正确召回；
显存峰值 72,758 MiB（共 73,415），余量 0.64 GiB，主机最低可用 14 GiB，无 OOM（[数据](results/capacity-522k-20261019.json)）。

## 10-19 优化内容

各项改动都可以用环境变量单独关闭（设为 0 即回到 10-18 的路径），lkqmoe 内核与编译库未改动。

| 改动 | 开关（默认） | 作用 |
|---|---|---|
| prefill 索引器 | `DS41_DENSE_FP4_INDEXER=1` | 1×/2× 压缩层的索引器原先在长 prefill 时退回 torch 参考实现（逐列 bf16 打分，开销随“块长 × 历史长度”增长，500K 时约占每个 32K 块的 70%）。现在改走 DeepGEMM FP4 MQA logits + 融合 top-k，按行分块控制显存。与参考实现的 top-512 重合度 97–99.5%，差别来自参考实现的 bf16 舍入；decode 原本就走同一个 DeepGEMM 内核。 |
| decode 验证同步 | `DS41_C128_HOST_BOUND=1` | DSpark 验证步原先为选 c128 CUDA graph 读取 GPU 上的序列长度（D2H 同步），主机必须等上一步跑完才能准备下一步。现在用调度器已知的请求长度加余量作为上界，只会选同宽或更宽（结果一致）的图；校验 2000 步无一次低于实际值。 |
| 草稿词表头 | `DS41_DRAFT_LMHEAD_FP8=1` | 草稿每步要读一遍共享的 bf16 词表头（1.32 GB）。改为按行缩放的 FP8 副本（多占 0.66 GB），该矩阵乘 1.11 → 0.62 ms。只影响草稿提案，所有输出 token 仍由目标模型的 bf16 词表头验证。 |
| 热专家换入节奏 | `LKQSTRATA_EVERY=2`、`LKQSTRATA_SWAPS=32` | 原为 4 / 16。决策更频繁、每窗口可换入更多，热专家命中率约 0.51–0.57 → 0.54–0.61。 |
| Engram 提前查表 | `DS41_ENGRAM_PREFETCH=1` | Engram 第 1、14 层的行原先在走到该层时才同步查（拷 id 回主机、主机回调查缓存或读 Optane、拷回），每步 GPU 空等约 2.3 ms。哈希在每步开头就已算好，现在两层的查表在哈希后立刻各走一条侧流，与第 0 层（第 14 层则是第 2–13 层）的计算重叠；取到的行与原来完全相同，wkv 投影仍在主流。 |
| 分块与热专家槽 | `BATCH_TOKENS=22528`（槽数按分块自动选：22528 → 50） | prefill 临时显存随分块大小变化（500K 上下文时每 1K 分块约 0.5 GiB），省出的显存放热专家：每层 43 → 50 槽。22528 是让 64K 输入切成 3 块的最小整页分块（每块有约 3 秒与块大小无关的专家上传开销）。 |
| prefill 软件流水 | `PREFILL_PREFETCH_CHUNKS=6`（分块 22528 时默认，0 关闭） | lkqmoe 自带的流水：FP4 解包和上传各走独立流，本层最后几块一解包完就开始上传下一层的前 6 块，与下一层注意力重叠。逐位一致；prefill 约 +7~9%，522K 峰值多约 0.75 GiB。 |
| Engram 并行读 | `DS41_ENGRAM_IO_THREADS=16` | 行存储原先每 32 行做一轮异步读，读盘实际是串行的（新文档 93K prefill 每层约 4 万轮）。现在一次查询的所有未命中合并、由 16 个常驻线程并发 O_DIRECT 读。新旧实现在真实 Engram 表上逐字节对照一致；20 万行查询 1.25 → 0.42 秒。 |


## 运行方式

OpenAI消息入口支持 developer 角色：编码前按原顺序映射为模型原生 system，保留指令、文本块和工具字段；聊天、思考模式及已有上下文均兼容。此处理在历史思考内容裁剪之前完成，避免静默丢失应用指令。

目标专家保留紧凑的 MXFP4 权重和 UE8M0 block-32 scale，CPU 与 GPU 热专家池同时计算；
长预填充按专家分块上传 GPU。共享专家、注意力和索引器在 GPU。
默认 static 全验证，不构建置信度头；生产启动不采集 STS 样本。
保留 `VERIFY_MODE=cap-sts` 备选，只有选择时才加载 `config/dspark-sts-calib.json`；
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
CHECK_ONLY=1 bash launch/run-ds41-1017.sh
bash launch/run-ds41-1017.sh
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

### API角色回归

设置 LSGLANG 指向已打补丁的源码目录后，可运行 pytest tests/test_developer_roles.py。现有编码样例和新增角色测试共77项通过；真实网关的developer、系统加developer、多轮流式与Responses四类请求通过，另有一次本机LAN流式请求通过。源代码更新在服务启动时加载，已有进程可以通过网关角色兼容恢复调用。
