# DeepSeek V4.1 Flash 1017 · Lsglang + lkqmoe

官方 **DeepSeek-V4.1-Flash-1017（MXFP4）** 单卡推理框架：RTX PRO 5000 72GB、双路 EPYC，
**512K 共享 KV、最多 2 并发、DSpark、默认变长验证**，附编译好的 **lkqmoe 0.5.6**。
本页参数和速度来自 2026-10-20 的部署：在 10-19 的配置（prefill 分块 22528、每层 50 个热专家槽、prefill 软件流水、lkqmoe CPU Down 一次读一行）上，DSpark 改为变长验证（每步按置信度验证 1–6 个 token，两个请求同时 decode 时 +14%），lkqmoe 0.5.6 的 GPU prefill 改用 FP4 张量核（prefill +6–10%）。
速度统一用编码代理测速 `bench/agent_bench.py`（多步工具调用，5 个会话从 64K 到约 490K，覆盖 512K KV 池），见下文。

框架基于 [guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)；
lkqmoe 受该作者的 lk_moe CPU MoE 后端启发，是独立实现的通用 CPU/GPU MoE 内核，不包含 lk_moe 实现代码。
核心源码保留在私有仓库，本仓库提供编译包及开源框架适配。

English: DeepSeek V4.1 Flash 1017 MXFP4 on one GPU, with 512K shared KV, two concurrent
requests, GPU-resident DSpark and the compiled lkqmoe 0.5.5 backend. The 2026-10-19 update keeps
long prefills on the DeepGEMM FP4 indexer and takes the Engram row lookups off the decode critical path; the midday
update uses a 22528-token prefill chunk, 50 hot-expert GPU slots per layer and lkqmoe's prefill pipeline; the evening
update ships lkqmoe 0.5.5 (one-row CPU Down). The 2026-10-20 release verifies 1-6 DSpark tokens per request from this step's confidence (variable-length verify, the default; +14% with two requests decoding) and runs the GPU prefill experts on the SM120 FP4 tensor cores (lkqmoe 0.5.6, prefill +6-10%). Speeds come
from the coding-agent benchmark `bench/agent_bench.py` (multi-step tool use, five sessions from 64K to ~490K context).

## 组成

| 部分 | 当前版本 / 内容 |
|---|---|
| 模型与草稿 | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)，revision `1bae0e1088993b9b82de08f59de5be5a34e42170`；草稿从同一检查点提取 |
| Lsglang 基线 | `6068667581bebd2184f7e4acb46136f44837889a` |
| 1017 框架补丁 | `patches/lsglang-6068667-1017.patch`（14 个文件），含官方会话编码、block-32 激活量化、Engram 与 DSpark 调整，以及 10-19 的 decode 验证免同步和草稿词表头 FP8 |
| 运行时覆盖 | `overlay/`（含 10-19 的 prefill 索引器分块）；Engram 行存储（10-19 起提前查表、并行读）、量化和按层提前加载在 `adapter/` |
| lkqmoe | 0.5.6（10-20：GPU prefill 走 FP4 张量核 `LKQMOE_PREFILL_FP4`；动态库与 0.5.5 相同，AVX512 库用 `-DLKQMOE_DOWN_R4=1` 编译）；源码提交 `93edcc19eb06a71bccf8e8974d8c8c28decda344`；4 个动态库、Python 3.12 字节码、GPU 适配和热专家排名种子 |
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
| DSpark | 3 层草稿常驻 GPU；块大小 5、每步验证 1–6 token（变长验证，按当前步置信度定长）|
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

5 个会话的起点对齐原来的五点测速（约 64K / 128K / 256K / 384K / 480K），覆盖整个 512K KV 池：
- 每个会话先对整段 prompt 做一次冷 prefill，之后多步调用工具，上下文继续增长。
- 下一次请求可能超出 524288 时，会话提前结束。
- 每会话最多 24 步，采样用服务端默认（同 dsh，不指定温度）。
- 模型写出的代码只做语法检查，不会执行。

负载形态参照 10-19 一次真实 dsh 会话：77 个请求，前缀缓存命中 98.6%，大多是工具调用，夹杂长的写文件输出。

| 会话 | 上下文（起 → 止） | 冷 prefill：首字 s / tok/s | 请求数 | 输出 token | decode tok/s | DSpark 接受长度 | 每步 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 64k | 66K → 106K | 18.9 / 3,516 | 24 | 6,186 | 143.0 | 4.74 | 33.0 |
| 128k-write | 132K → 157K | 32.0 / 4,132 | 24 | 3,766 | 136.3 | 4.43 | 32.3 |
| 256k | 265K → 401K | 63.9 / 4,141 | 24 | 1,191 | 195.2 | 5.37 | 26.9 |
| 384k | 394K → 417K | 99.1 / 3,978 | 18 | 4,606 | 147.2 | 5.03 | 34.0 |
| 500k | 489K → 518K | 127.3 / 3,844 | 19 | 5,856 | 135.3 | 4.68 | 34.5 |
| **全部** | | | 109 | 21,605 | **142.5** | 4.75 | 33.2 |

- 命中前缀缓存的后续步骤，首字中位 **2.05 s**。
- 本轮是 10-20 的生产配置（变长验证 + FP4 prefill）；变长验证对单请求和两个请求的作用见下文"10-20 优化内容"。
- 输出长度 p10 / p50 / p90 为 47 / 77 / 479 token；≥ 500 token 的长输出（写文件、改代码）占输出 token 的 52%。
- 计算口径：
  - 冷 prefill tok/s = 未命中缓存的 prompt token ÷ 首字时间。每轮测速的系统提示带有本轮编号，不会命中以前的缓存；同一轮的会话之间只共享约 1–2K token 的系统提示和工具定义。
  - decode tok/s = 输出 token ÷ 首末流式片段间隔。
  - 接受长度 = 输出 token ÷ DSpark 验证步数；每步 ms = decode 时间 ÷ 验证步数。
- 接受长度随内容和采样变化，同一配置两次运行约差 ±0.4。比较配置时先看"每步 ms"，decode tok/s 是这类任务上的实际体感。
- 旧的五点、多领域 decode 和并发测速已从仓库移除，统一用本测速。

### lkqmoe 0.5.5 前后（2026-10-19 晚）

上表是 0.5.5 的结果；0.5.4 时为 decode 143.1 tok/s、每步 33.8 ms。两次运行的内容、接受长度和各会话的输出量都不同（每步耗时随上下文长度变化），所以 agent_bench 不用来比较内核。
内核本身用同日、同一组多领域提示（20 个 × 温度 0 / 0.7，单请求）对比，先测新版再换回旧版：

| 温度 | 0.5.4 每步 ms | 0.5.5 每步 ms | decode tok/s |
|---|---:|---:|---:|
| 0 | 32.6 | 31.5 | 97.8 → 99.8 |
| 0.7 | 32.4 | 31.5 | 96.1 → 98.2 |

- 原因：DeepSeek V4.1 的 Down 权重每行只有 1152 字节，原内核一次交错读 4 行，读内存比 Gate 慢约 30%；改为一行一条流，并把动态领取粒度设为 24 行后，CPU Down 快 15% 到 17%，整次 CPU 专家调用快约 6%。输出逐位不变。
- 每次调用的耗时（按 token 数分）见 `results/lkqmoe-down-ab-20261019.json`。

[原始数据](results/agent-bench-20261020.json)

**满容量验证（不是测速）**：522,240 输入 + 1,024 输出（KV 池 524288，变长验证 + FP4 prefill）完成且正确召回，首字 137.1 s；
显存峰值 72,816 MiB（共 73,415），余量 0.58 GiB，无 OOM（[数据](results/capacity-522k-20261020.json)）。

## 10-21：分词与官方 deepseek-recipe 一致

- 原来 V4.1 的 prompt 渲染成字符串后整串分词，用户消息、工具结果、工具定义里出现的 `<｜User｜>`、`<｜end▁of▁sentence｜>`、`</think>` 等写法会被编成真正的控制 token（模型会看到伪造的对话边界，读 DeepSeek 相关源码或模板时就会触发）。
- 现在按 DeepSeek 官方 [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) 的规则：只有模板插入的控制 token、图片占位符和 assistant 正文 / 思考里的 grounding token 是特殊 token，其余内容用去掉 added vocabulary 的 tokenizer 按普通文本编码，各段一批编码（`encoding_dsv41.encode_messages_to_ids`；`DS41_PLAIN_CONTENT_TOKENS=0` 回到原路径）。
- 验证（`results/plain-content-tokens-20261021.json`）：40 组普通对话的 token ID 与原路径逐个相同；6 组请求（含注入、grounding、工具结果、思考模式）与 deepseek-recipe 0.2.0 逐 token 相同；长会话命中缓存的轮次（新增 < 2K token）首字 400–600K 上下文 2.20 → 1.81 s。

## 10-20 优化内容

| 项目 | 开关（启动脚本默认） | 说明 |
|---|---|---|
| DSpark 变长验证 | `VERIFY_MODE=compact` | 每步按草稿置信度验证 1–6 个 token（思路来自 DeepSeek DeepInfer 的 `mtp_varlen`）。需要：Engram 哈希器支持变长行；验证图的填充 token 不进 CPU 专家（`LKQMOE_ACTIVE_FLAGS=1`）；逐点"验证 token 数 → 步耗时"表（`overlay/sps-table-compact.json`）；设备端用当前步置信度定长（`DS41_DSPARK_DEVICE_BUDGET=1`，原规划器用两步前的置信度，单请求反而 −2%）。`static` / `cap-sts` 仍可选。 |
| GPU prefill 走 FP4 张量核 | `PREFILL_FP4=1`（`LKQMOE_PREFILL_FP4`） | lkqmoe 0.5.6：prefill 专家用 FlashInfer 0.6.18 CUTLASS fused MoE（MXFP4 权重 × MXFP8 激活，即官方参考的激活精度），不再解包成 BF16；热专家池里的专家走显卡内拷贝，暂存 12 槽，每次调用最多 4096 token。prefill 的 MoE 本来就受 PCIe 上传限制，所以收益来自更大的预取环和热池复用。 |

变长验证（同日、同一组多领域提示，`results/varlen-verify-20261020.json`）：

| 场景 | static | 变长验证 |
|---|---:|---:|
| 单请求，温度 0 | 101.33 tok/s | 104.42 tok/s |
| 单请求，温度 0.7 | 99.19 | 98.41 |
| 两个请求同时 decode（合计） | 100.1 | 114.03 |

- 单请求每步便宜约 5%，但每步产出也少一些，净 +1% 到 +3%（日间波动约 ±2%）。两个请求时，总验证 token 超过 6 就要换 12 token 的验证图（贵约 10 ms），变长验证多数步留在 6 token 档，所以收益大。

FP4 prefill（首字 s，五点 + 522K，同日，都开变长验证，`results/prefill-fp4-20261020.json`）：

| prompt | BF16 路径 | FP4 路径 | 提升 |
|---|---:|---:|---:|
| 64K | 18.4 | 16.8 | +9.7% |
| 128K | 32.5 | 30.0 | +8.6% |
| 256K | 66.8 | 61.6 | +8.5% |
| 384K | 104.2 | 96.7 | +7.7% |
| 500K | 139.8 | 131.8 | +6.1% |
| 522K | 145.9 | 137.1 | +6.4% |

- 精度（256 token，相对 L2）：FP4 路径对官方参考 1.7%，对 BF16 激活参考 4.5%，与官方量化本身（4.5%）相同。显存峰值与 BF16 路径相同。

评估过、没有采用：DeepSelect top-k（SM120 上只有 prefill 2048×256K 快 1.65×，decode 不快，整体约 0.5%）；DeepInfer 的 TileLang mHC（mHC 每步约 1.2 ms GPU，用放慢法实测只有约 0.4 ms 在关键路径上，移植上限约 1%）。

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
默认 `VERIFY_MODE=compact`（变长验证），加载 `config/dspark-sts-calib.json` 和 `overlay/sps-table-compact.json`；生产启动不采集 STS 样本。
`VERIFY_MODE=static`（固定验证 6 个）和 `cap-sts` 仍可选，`cap-sts` 也加载同一校准文件；
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
