# DeepSeek V4.1 Flash 1017 · Lsglang + lkqmoe

官方 **DeepSeek-V4.1-Flash-1017（MXFP4）** 单卡推理框架：RTX PRO 5000 72GB、双路 EPYC，
**512K 共享 KV、最多 2 并发、DSpark、默认 static 验证**，附编译好的 **lkqmoe 0.5.4**。
本页参数和速度均来自 2026-10-18 的 1017 部署。

框架基于 [guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)；
lkqmoe 受该作者的 lk_moe CPU MoE 后端启发，是独立实现的通用 CPU/GPU MoE 内核，不包含 lk_moe 实现代码。
核心源码保留在私有仓库，本仓库提供编译包及开源框架适配。

English: DeepSeek V4.1 Flash 1017 MXFP4 on one GPU, with 512K shared KV, two concurrent
requests, GPU-resident DSpark and the compiled lkqmoe 0.5.4 backend.

## 组成

| 部分 | 当前版本 / 内容 |
|---|---|
| 模型与草稿 | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)，revision `1bae0e1088993b9b82de08f59de5be5a34e42170`；草稿从同一检查点提取 |
| Lsglang 基线 | `6068667581bebd2184f7e4acb46136f44837889a` |
| 1017 框架补丁 | `patches/lsglang-6068667-1017.patch`，含官方会话编码、block-32 激活量化、Engram 与 DSpark 调整 |
| 运行时覆盖 | `overlay/`；Engram 行存储、量化和按层提前加载在 `adapter/` |
| lkqmoe | 0.5.4；源码提交 `9cbeb6c7a8a3bd2fc36ced525004450eb3a1dee5`；4 个动态库、Python 3.12 字节码、GPU 适配和热专家排名种子 |
| 启动入口 | `launch/run-ds41-1017.sh` |
| 依赖 | `requirements-lock.txt`；附本次使用的 FlashInfer 0.6.18 和 b12x 1.3.0 轮子 |
| 实测与完整性 | `results/`、`SHA256SUMS` |

## 本机硬件与参数

| 项目 | 实测配置 |
|---|---|
| CPU | 2× EPYC 9334，64 核 / 128 线程，NPS2、4 个 NUMA 节点 |
| 内存 / GPU | 320 GB DDR5；RTX PRO 5000 72GB Blackwell（SM120） |
| 模型存储 | 三星 990 PRO；Engram 位于低延迟 Optane NVMe |
| 主模型专家 | 0 层整层常驻；40 层各使用 CPU + 43 个 LKQstrata GPU 热专家槽，自适应替换 |
| DSpark | 3 层草稿常驻 GPU；块大小 5、每步验证 6 token；static |
| KV / 并发 | 524288 token，共享池最多 2 请求 |
| 预填充分块 / 显存比例 | 32768 / 0.95 |
| Engram / CPU | 行缓存 4 GiB；112 线程、BF16 Down、动态调度 |
| 预填充切换阈值 | hybrid 上限 6144，CPU 阈值 1024 |
| 其他 | FP8 KV、FP4 indexer、SWA 尾部有界重放 |

两个请求共享 512K KV 池，不是每个请求各有 512K。
每层 CPU 专家加载完成后立即交给 lkqmoe 并释放临时 scratch；`DS41_EARLY_LK_MOE=1` 默认开启。
本机一次完整 static 启动约 **310 秒**；权重加载记录约 **200 秒**，耗时会随页缓存和存储状态变化。

## 512K 容量内五点测速

温度 0，每点输出 1024 token，独立长会话文档，事实埋在约 10% 深度。
**5/5 正确召回，全部输出 1024 token，无请求错误。** 表内是实际输入长度；最大点是 499961 输入，
本轮没有把该点描述为精确填满 524288。

| 实际输入 token | 首字时间 s | prefill tok/s | decode tok/s | 接受长度 |
|---:|---:|---:|---:|---:|
| 65,524 | 17.90 | 3,661 | 90.9 | 3.58 |
| 131,076 | 44.25 | 2,962 | 78.1 | 3.42 |
| 262,113 | 98.53 | 2,660 | 85.8 | 3.48 |
| 393,206 | 173.67 | 2,264 | 78.4 | 3.17 |
| 499,961 | 252.07 | 1,983 | 78.6 | 3.27 |

**prefill = 实际输入 token ÷ 首字时间**，包含请求、调度、tokenization 与首 token 等开销，非纯 GPU 内核速度。
本打点脚本 decode = 输出 token ÷ 首内容至流结束的时间；接受长度采用本请求原生统计。
[五点原始数据](results/speed-points-20261018-1017-512k.json)。

## 64K 输入＋1K 输出：1 / 2 并发

每请求严格 65536 输入、1024 输出，温度 0，各两轮，缓存命中均为 0。
双并发同时提交，已确认两个请求的解码时间实际重叠。

| 并发 | 平均首字 s | 每请求 prefill tok/s | 每请求 decode tok/s | 合计生成窗口 tok/s | 端到端总输出 tok/s | 接受长度 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.53 | 3738 | 77.4 | 77.5 | 33.3 | 3.11 |
| 2 | 30.41 | 2205 | 40.1 | 63.4 | 35.2 | 3.08 |

每请求 decode 按首末生成 token 的时间与 token 增量计算。
“合计生成窗口”从本轮最早首 token 到最晚末 token，双并发包含另一请求的预填充等待；
“端到端总输出”还包含最初的预填充。因此不能把这两列与每请求速度直接相加比较。
两种负载的端到端总输出约为 **33.3 / 35.2 tok/s**，没有双并发解码成倍提升的结论。
五点与并发使用不同文档种子，两个表的 64K 数值并非同一请求。
[并发原始数据](results/concurrency-64k1k-20261018.json)。

## 运行方式

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
"$VENV/bin/python" bench/speed_points.py --port 39504   --model DeepSeek-V4.1-Flash-1017 --tokenizer "$MODEL"   --points 65536,131072,262144,393216,500000 --output 1024   --exact --stop-on-fail --out results/new-five-points.json
PYTHONPATH="$LSGLANG/python" "$VENV/bin/python" bench/concurrency64k.py   --model-path "$MODEL" --port 39504 --rounds 2   --input-tokens 65536 --output-tokens 1024 --out results/new-concurrency.json
```

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
