export function taskPlanLines(plan) {
  if(!["quoted-constraints-static-plan-v1","quoted-constraints-static-plan-v2-source-ids"].includes(plan?.version))return [];
  const kinds={lookup:"查询",calculate:"计算计划",synthesize:"汇总计划"};
  const states={planned_lookup:"已形成静态检索线索",waiting_on_evidence:"待依赖证据，未执行",
    not_separately_executed:"未独立执行，由完整问题的回答阶段处理"};
  return [
    `任务目标（模型提炼）：${plan.goal}`,
    ...((plan.constraints||[]).map(c=>`条件 ${c.id} · ${c.scope==="global"?"全局":"局部"} · 用户原话 ${c.source_id}：${c.quote}`)),
    ...((plan.steps||[]).map(s=>`${s.id} · ${kinds[s.kind]||"类型未知"}：${s.question}；依赖：${s.depends_on?.join("、")||"无"}；条件：${s.constraint_refs?.join("、")||"未标注"}；${states[s.status]||"状态未知"}。`)),
    `共享检索：${plan.retrieval_status==="shared_retriever_returned"?"已返回候选（不代表每步都有证据）":"尚未记录返回"}；候选总数 ${Number.isFinite(plan.shared_candidate_count)?plan.shared_candidate_count:"未知"}。`,
    "最多两个独立lookup与原查询进入同一候选池，不增加原召回/上下文上限。依赖查询不猜值、不动态执行；没有逐步答案、计算器或完成率。条件出处通过字面校验，不证明提取完整或语义正确；回答仍以完整用户原话和文档证据为准。",
  ];
}
