const stages={context_resolution:"上下文解析",query_rewrite:"查询改写",task_planning:"任务规划",query_embedding:"查询向量化",
  referent_candidates:"对象候选识别",rerank:"重排",relevance_assessment:"相关性检测",answer_generation:"首稿生成",
  answer_review:"依据审核",answer_repair:"纠错生成"};
const causes={deadline_exhausted:"整轮时限已到",provider_calls_exhausted:"调用次数用完",
  reserved_tokens_exhausted:"累计预留token额度不足",output_reservation_missing:"缺少明确输出预留",
  provider_error:"调用未完成",cancelled:"请求中断"};
const n=value=>Number.isFinite(value)?String(value):"未知";

export function roundBudgetLines(budget) {
  if(budget?.version!=="online-round-admission-v1")return [];
  return [
    `整轮执行预算：已尝试 ${n(budget.attempted_calls)} / ${n(budget.call_limit)} 次，阻止 ${n(budget.blocked_calls)} 次；累计预留/补计 ${n(budget.committed_reserved_tokens)} / ${n(budget.token_reservation_limit)} token。`,
    `状态：${budget.status==="stopped"?"已触及限制":"未触及限制"}；耗时 ${n(budget.elapsed_ms)} ms / 时限 ${n(budget.deadline_seconds)} 秒；实报总token ${n(budget.observed_total_tokens)}，已知小计 ${n(budget.known_reported_tokens_subtotal)}，用量未知 ${n(budget.usage_unknown_calls)} 次。`,
    ...((budget.calls||[]).map(row=>`${stages[row.stage]||"未分类阶段"} · ${row.kind==="chat"?"语言模型":row.kind==="embedding"?"向量":row.kind==="rerank"?"重排":"类型未知"}：${row.status==="blocked"?"已阻止，未发送":row.status==="response_received"?"已收到响应":row.status==="cancelled"?"已中断":row.status==="failed"?"未完成":"执行状态未知"}；预留 ${n(row.reserved_tokens)} token；${causes[row.reason]||"无阻止原因"}。`)),
    "预留是本地估算输入加输出上限，不随低用量或失败退还；实报高于预留则补计。各供应商分词不同，缺失用量不是零，未计算账单金额；收到响应不代表答案正确。在线SDK自动重试已关闭，超时不保证供应商未计费。",
  ];
}
