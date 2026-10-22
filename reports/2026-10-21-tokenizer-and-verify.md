# 2026-10-21：分词与官方 deepseek-recipe 一致；放弃 DSpark 变长验证

## 分词

- 原来 V4.1 的 prompt 渲染成字符串后整串分词，用户消息、工具结果、工具定义里出现的 `<｜User｜>`、`<｜end▁of▁sentence｜>`、`</think>` 等写法会被编成真正的控制 token（模型会看到伪造的对话边界，读 DeepSeek 相关源码或模板时就会触发）。
- 现在按 DeepSeek 官方 [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) 的规则：只有模板插入的控制 token、图片占位符和 assistant 正文 / 思考里的 grounding token 是特殊 token，其余内容用去掉 added vocabulary 的 tokenizer 按普通文本编码，各段一批编码（`encoding_dsv41.encode_messages_to_ids`；`DS41_PLAIN_CONTENT_TOKENS=0` 回到原路径）。
- 验证（`results/plain-content-tokens-20261021.json`）：40 组普通对话的 token ID 与原路径逐个相同；6 组请求（含注入、grounding、工具结果、思考模式）与 deepseek-recipe 0.2.0 逐 token 相同；长会话命中缓存的轮次（新增 < 2K token）首字 400–600K 上下文 2.20 → 1.81 s。

## 放弃 DSpark 变长验证

- 变长验证（`VERIFY_MODE=compact`）是 10-20 上线的：每步按草稿置信度验证 1–6 个 token，做法与测速见 [2026-10-20 报告](2026-10-20-fp4-prefill.md)。
- 单请求没有收益（+1–3%，在日间波动内）；两个请求 +13.9%，但两种情况都会改输出 token：温度 0 照抄测试中，单请求 4/8–6/8 段出错、两个请求约 1/8 出错，
  固定验证在 28 段单请求、12 段两请求中都没有中间错字。分歧从第 1 层（只有 SWA 的层）的注意力输出开始，根因未找到。
- 收益抵不过风险，不再继续迁移：启动脚本拒绝 `VERIFY_MODE=compact`，变长验证的逐点成本表已删除，`RELEASE.json` 记在 `dropped_features`。DSpark 固定验证 6 个 token。
