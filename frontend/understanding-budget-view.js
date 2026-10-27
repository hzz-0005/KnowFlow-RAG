export function understandingBudgetLines(budget) {
  if (budget?.version !== "understanding-whole-request-v1") return ["未记录本阶段完整请求预算，保持未知。"];
  const n=value => Number.isFinite(value) ? String(value) : "未记录";
  return [
    `完整请求预算：输入估算 ${n(budget.input_tokens_estimate)} token；输出预留 ${n(budget.output_reserved_tokens)}，安全余量 ${n(budget.safety_tokens)}；合计 ${n(budget.estimated_total_tokens)} / 运行目标 ${n(budget.configured_total_tokens)}。`,
    `预算判定：${budget.estimated_feasible === true ? "估算可容纳" : budget.estimated_feasible === false ? "预算不足，未调用模型" : "未知"}；实报输入 ${n(budget.provider_usage?.prompt_tokens)}、输出 ${n(budget.provider_usage?.completion_tokens)} token；缺少用量不填零。`,
    "完整问题与条件不截断；本地分词及运行目标不是供应商真实窗口，不证明理解正确，也不是整条查询的累计预算。",
  ];
}
