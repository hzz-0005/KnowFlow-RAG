# 查询改写：执行诊断与边界

2026-10-24，查询契约 `query-only-multilingual-scope-v3-protected-numbers`，诊断契约 `query-rewrite-diagnostics-v1`。v2 是本轮诊断中间版，真实失败记录仍保存其版本。

## 用途及信息流

混合/重排混合/图混合查询可额外生成最多两个检索式。原问题仍参与召回；改写不替代原始用户条件，也不提供文档事实。会话指代解析是此前另一个步骤，不与本步骤混为一谈。未启用或纯向量/关键词查询不执行本步骤的模型调用。

本轮保留每条检索式的数字集合校验，新增响应完整性检查与分阶段记录。实际复测暴露 numeric_constraints_changed 后，将从原问题机械提取的数字字面值列表作为 protected_numeric_tokens 随完整原问题提供；提示明确保留文件编号、前导零、标识符及小数的原始角色，不能随意附一串数字过检。不能忠实改写时返回原问题，不自动补写模型漏掉的数值。

与前版相比，`finish_reason != stop` 的内容一律不用，即使它恰巧是合法 JSON。没有新增自动重试、模型切换、放宽数字规则或用户数据迁移。显式数字输入仍只是提示约束，是否改善不同场景需独立评测；不保证不会再次拒收。

## 执行记录

`QueryResponse.query_plan` 随原有响应、会话轮次和质量评测轨迹保存。字段是附加字段，无需数据库迁移；旧轮次按旧数据读取，不回填推测的原因。

| 字段 | 含义 |
|---|---|
| `status` | `disabled / complete / unavailable`；策略未请求时原调用方返回 `not_requested` |
| `phase` | `configuration / request / response / validation / complete` |
| `reason_code` | 单值机械原因，见下表；不是语义正确标签 |
| `request_attempted` | 已进入 SDK 请求；不能证明远端接收或计费 |
| `response_received` | SDK 已返回可访问响应对象；不保证内容有效 |
| `fallback_used` | 启用后失败而只用原问题；与未启用、有效但未新增检索式分开 |
| `original_only` | 本步骤没有接纳额外检索式；原问题依然参与查询 |
| `provider_usage` | 供应商实报的非负整数输入/输出/总 token；缺失为 null，不填零 |
| `response_content_sha256 / response_content_chars` | 已有字符串正文的指纹及字符数，非法/截断文本同样可有指纹；不另存原始失败正文 |
| `finish_reason / http_status` | 归一化结束标记、SDK HTTP 状态；无原始报错正文、请求头或 URL |
| `protected_numeric_token_count / request_messages_sha256` | 原问题中不同数字字面值数量、实际提示消息指纹；不是从语料或历史答案提取 |
| `validation_failure` | 数字不符时的 1 基检索式序号、遗漏/新增的不同数字数量；没有保存非法检索式全文 |
| `latency_ms` | 本步骤总耗时；不当成原因判断依据 |

阶段与原因：

- 配置：`client_configuration_error`；未发请求。
- 请求：`local_timeout / provider_timeout / provider_rate_limited / provider_http_error / provider_connection_error / provider_error`。
- 响应完整性：`empty_choices / output_limit / content_filtered / finish_reason_missing / unexpected_finish_reason`。
- 内容校验：`empty_content / invalid_json / invalid_schema / invalid_query_count / invalid_query_text / numeric_constraints_changed`。
- 完成：`accepted`，或 `no_effective_queries`（去重后没有不同于原问题的检索式）。后者不是失败，也不表示扩展成功。
- 未分类程序异常：`unexpected_error` 加当前阶段；不能伪装成已知供应商故障。

仍沿用一次 SDK 调用、`max_retries=0` 及本地改写时限。外部取消传播给上层，不被包装成普通回退。缺少 usage 可能仍有实际计费，不能据此断言无成本。失败信息用封闭原因码，不把异常字符串、接口响应、密钥或任意服务端字段带到页面。

## 校验能说明什么

每条检索式要求 JSON 顶层对象、1–2 条字符串、每条去空白后 4–800 字符、数字集合与原问题逐字相同。任一条非法就整组拒收，不返回此前已处理的合法条。数字来自文件名、编号也在现有规则内；这可能过严，但本轮不为单一开发问题降低保护。

集合校验不能发现数字对调、单位变化、否定反转、实体和时间范围改变。`accepted` 只说明这些机械检查通过，不能写成“语义保真通过”。改善该能力需要独立标签和通用保护机制，留在 C2；不能用两个检索式或格式通过来宣称长问题拆解完成。

## 历史与可比性

旧记录仅有 `unavailable + latency`，具体原因未知。当前再次调用的观测不能证明历史调用也是同一错误。旧页面展开时明确提醒这一点，不改写旧轮次。

新质量报告保存 `query_rewrite_contract`：版本、提示 SHA-256、数字校验版本、数字保护输入协议、完成策略、启用开关、时限与重试数。新增与历史缺失、或不同契约的报告不能硬判为同策略回归。两份均无该字段的历史报告保留原有行为，不因此宣称其历史协议已齐全。M1 冻结账本没有修改。

## 验收

自动检查覆盖空响应、非法 JSON/结构/列表/长度、数字遗漏或新增、整组原子拒收、去重、原样输出、正常结束与截断、真实本地计时取消、供应商超时/HTTP/限流/连接异常、密钥字符串不泄漏、外部取消传播、空检索仍保存原因、历史页面兼容及安全渲染。真实侧边栏复测与工程数量以 本包记录（本地研究记录，未随公开包分发） 为准，均不代表泛化质量成绩。

## 外部依据

参照 [百炼结构化输出](https://help.aliyun.com/en/model-studio/qwen-structured-output)：JSON Object 不保证目标 schema，故仍需本地校验；[Microsoft JSON mode](https://learn.microsoft.com/azure/ai-services/openai/how-to/json-mode?tabs=python) 说明先检查截断状态；[LangChain structured output](https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel/with_structured_output) 将解析结果与解析异常分开。适配与社区线索见 research.md（本地研究记录，未随公开包分发），未引入 LangChain 依赖。
