# 上下文解析与查询改写的完整请求预算

2026-10-25，`understanding-whole-request-v1`。本包在 C2 上补两个阶段的调用边界，不扩展会话记忆范围。实现为 `understanding_budget.py`、`conversation_context.py`、`query_plan.py`。

原上下文 payload 上限仍为3000本地token：整条用户原话、已引用的原始条件、待澄清补充及来源身份必须完整装配；最近必需意图装不下，不能用更早短话题替代。这个上限只约束载荷，不包含系统提示与输出。因此在实际 SDK 调用前另检查完整 messages：

`input_estimate(messages) + output_reserved + safety <= configured_total`

输入估算包含两个角色、完整系统提示与序列化载荷、估计的聊天封装开销。沿用本地分词器并记录SHA；供应商分词匹配仍 unverified。估算不包含 HTTP 字段原文作为 token，但完整内容与 JSON 标记已计入；运行目标不是供应商模型窗口保证，也不能预言最终真实输出数量。

| 配置前缀 | 总预算 | 输出预留 | 安全余量 | 输出参数 |
|---|---:|---:|---:|---|
| CONTEXT_RESOLUTION | 8192 | 2048 | 512 | auto |
| QUERY_REWRITE | 8192 | 2048 | 512 | auto |

各前缀支持 `_TOTAL_TOKEN_BUDGET`、`_OUTPUT_TOKEN_BUDGET`、`_SAFETY_TOKENS`、`_OUTPUT_PARAMETER`。默认值为运行约束；本包不改实际.env或模型。显式输出参数可用auto/max_tokens/max_completion_tokens，沿用 `request_budget.output_limit` 的模型+端点适配；自定义代理语义未知。两阶段沿用关闭思考、零SDK重试、原本超时配置。

上下文超完整预算时零SDK调用，返回明确澄清和requires_restatement，不产生resolved intent；原记录仍在数据库，要求完整重述而非截断后暗自猜测。查询改写超预算则unavailable/phase=budget/fallback_used，原问题继续检索。无历史的独立首问、缺对象的冷指代或禁用改写沿用原零调用路径，未发生的预算不伪造成观测。

只有正常stop响应可进入原校验；length即使JSON合法也不采用。数字集合/原始条件传递/引用轮次校验保持，不能保证模型在语义上保真。每阶段request_budget记录完整消息SHA、配置/估算、是否发起/收到、结束原因、实报usage；无用量为null，真实0保留。异常只用封闭原因码，不保存供应商异常正文或凭据。外部取消不转成正常回退。

查询改写契约新增该预算及分词器配置，缺失或不同契约不能直接比较。上下文预算保存到逐轮resolution，详情“上下文判断/检索问题与原始条件”；改写预算保存至query_plan，详情“查询改写”。旧记录无字段显示未知，不从当前代码反推旧调用。

成熟接口依据：[百炼DashScope API输出参数](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-dashscope)与[Chat Completions结束原因/输出限额](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。借鉴接口约束，未引入框架依赖；原提示文本不变。

已覆盖首稿、相关性、审核/修复和本包两理解阶段的独立运行预算；后续[在线整轮预留首版](round-budget.md)已增加调用量/本地估算准入，仍无实际金额硬配额、摘要策略、任务依赖图、任务或长期记忆，也未建立多轮独立语义基线。工程检查和合成页面控制不能替代这些验收。实际证据见工作包（本地研究记录，未随公开包分发）。
