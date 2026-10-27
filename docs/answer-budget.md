# 首稿生成的上下文与输出预算

本包合同为 `answer-window-whole-sources-v1`。它统一计算**首稿生成这一请求**的固定提示、完整用户问题/继承条件、来源证据、输出预留和安全余量。查询解析/改写、相关性检测、独立依据审核与纠错生成仍各自调用模型，不宣称全链路预算已经统一，也没有新增摘要、任务记忆或长期记忆。

## 如何接纳来源

1. 复用 `providers.answer_messages` 构建实际发送的两条消息，避免预算程序与请求模板分别维护。
2. 使用现有离线 Qwen3 家族 tokenizer，按消息正文、角色和显式聊天封装估算开销。供应商精确 tokenizer/封装没有验证，不把估算称为真实消耗。
3. 首先原样保留用户任务，包括当前原话、明确引用的历史用户输入和更正。固定部分加输出预留已经超限时，在查询改写/召回之前明确拒答，不截断原始条件。
4. 相关性通过的完整来源先检查支持引文仍在该块内，再按原排名逐块试放。每次重新计算完整请求估算。整块不合额度就跳过，继续尝试后面的较小块；不会截掉半个表格、条件句或代码块来填满预算。
5. 只合并字面完全相同的重复块；大小写、缩进、空白差异不自行当作语义等价。每个预算输入块保留采用/跳过、原因和正文指纹；来源或引文检查失败与预算不足分开显示。
6. 发送模型前再用同一模板检查一次。返回 `finish_reason=length` 时不返回残缺正文、不自动放大预算或重试，保留状态供用户判断是否缩小问题。

拒绝/跳过的来源仍出现在原候选轨迹中，不能把未进入模型的来源当答案引用。整块策略可能放弃大而有用的来源；这是真实限制，不能据此保证证据完整。后续如增加有来源的句/表格行窗口，应另立合同并验证条件不丢失。

## 配置

无需填写新 key，现有供应商与凭据保持。`.env.example` 增加以下可选项，实际 `.env` 未改写：

| 配置 | 默认 | 含义 |
|---|---:|---|
| RAG_ANSWER_TOTAL_TOKEN_BUDGET | 32768 | 首稿请求的运行目标上限，不是模型公布的真实窗口 |
| RAG_ANSWER_OUTPUT_TOKEN_BUDGET | 8192 | 预留并请求供应商限制的输出额度；已核实的适配包含思考和正文 |
| RAG_ANSWER_SAFETY_TOKENS | 1024 | 分词/封装及供应商输出边界偏差的缓冲，不证明误差总在此范围内 |
| RAG_CONTEXT_TOKEN_BUDGET | 6000 | 来源证据在完整提示中的估算增量上限，含来源标签和分隔 |
| RAG_ANSWER_OUTPUT_PARAMETER | auto | 按受支持供应商选择输出字段；可明确配置字段，语义未核实则标未知 |

判据为：估算输入 + 输出预留 + 安全余量 ≤ 运行目标，且证据增量 ≤ 证据上限。更换模型时须核对实际窗口与参数能力；本包没有自动读取供应商窗口元数据。输出额度会限制复杂推理，触顶会明确拒答，不能宣称质量或速度必然改善。

`auto` 只对已核对的官方地址/模型族声明语义：百炼官方域名下当前 Qwen3.7/3.8、官方 MiMo2.6 使用 `max_completion_tokens`，DeepSeek 官方端点使用 `max_tokens`；仅凭模型名字不会把自建兼容地址判为已核实。其他兼容服务使用兼容字段但显示输出语义未核实，拒绝静默换模型或吞掉错误。

## 轨迹与展示

`QueryResponse.context_budget` 保存固定输入、证据增量、总估算、额度、tokenizer/问题/实际提示指纹、整块决策、模型路由、输出字段及语义。SDK 返回的 `usage` 单独保存为 `provider_usage`，未返回则为 null，不填 0；不保存思考正文。实报模型名不等于固定修订。

回答详情中的“首稿预算”展示本地估算与供应商实报，并说明首稿范围。超固定预算、没有来源可整块容纳、支持引文不符、无候选、输出触顶是不同状态。历史记录没有该字段时不补造预算。

2026-10-25 首稿模型选择单独接收内部 RoutingInput，分配依据为当前及引用的用户原话；预算仍测量完整实际提示，最终SDK复用同一个model_route。routing_input_sha256与prompt_sha256含义不同，前者不能证明完整提示已发送。复杂规则与完整任务不截断的约束不变；[路由契约与边界](model-routing.md)。

新质量评测报告保存 `answer_budget_contract`。额度、tokenizer 或整块策略不同，或旧报告缺此合同，普通同预算回归返回 not_comparable；不反推历史默认值、不改旧结果。M1 冻结合同与历史指标未改写。

## 外部依据及限制

- [LlamaIndex PromptHelper 源码](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/prompt_helper.py)：先扣提示与输出开销。本项目采用此思路，但为了保留表格/条件，采用整块接纳而非截前缀。
- [LangChain 短期记忆说明](https://docs.langchain.com/oss/python/langchain/short-term-memory)：消息历史需要预算管理。本项目不直接照搬“截最后 N 条”，已选原始用户条件作为固定任务保留。
- [百炼 Chat Completions 官方参数](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)：`max_tokens` 语义随模型变化，受支持模型的 `max_completion_tokens` 合计限制思考和正文，可能有小幅边界偏差。
- [MiMo 深度思考官方说明](https://mimo.mi.com/docs/en-US/quick-start/usage-guide/other/deep-thinking)、[DeepSeek Chat API](https://api-docs.deepseek.com/api/create-chat-completion/)：按各自协议使用字段，不假设兼容 API 的名字和计量都相同。

核对日期：2026-10-24。没有引入新框架或联网下载 tokenizer。测试、真实 SDK/页面验收、已知限制见 本包记录（本地研究记录，未随公开包分发）。
