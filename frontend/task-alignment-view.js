export function renderTaskAlignment(host, checks, onReview, open=false) {
  const rows=Array.isArray(checks)?checks:[];
  const latest=rows.at(-1), result=latest?.result;
  const expired=latest?.status==="running" && Number.isFinite(Date.parse(latest.lease_until)) && Date.parse(latest.lease_until)<=Date.now();
  const names={mismatch_detected:"模型发现任务偏离",no_mismatch_detected:"模型未发现偏离（不等于通过）",unknown:"结果未知"};
  const statuses={running:"正在核对",complete:"核对完成",partial:"部分条目未知",unavailable:"核对不可用",interrupted:"核对中断，结果未知",budget_exceeded:"完整输入超预算，未调用",not_applicable:"没有可核对的已完成回答"};
  const box=document.createElement("details");box.className="resolved-intent task-alignment";box.open=open;
  box.dataset.outcome=latest?.outcome || "unknown";
  const title=document.createElement("summary");title.textContent=`本轮任务核对：${latest ? names[latest.outcome] || "状态未知" : "尚未运行"}`;box.append(title);
  const line=text=>{const p=document.createElement("p");p.textContent=text;box.append(p);};
  line("与回答依据复核分开：检查是否答了当前目标、对象和范围。仅为模型诊断，覆盖未独立验证，不是任务成功率。");
  if(latest) {
    line(`记录 ${rows.length} 条；最近状态：${statuses[latest.status] || "未知"}。${latest.completed_at ? `完成于 ${new Date(latest.completed_at).toLocaleString()}` : ""}`);
    if(latest.input_snapshot?.goal_basis==="reconstructed_from_task_source_turn_only") line("旧轮次缺任务目标字段；仅从同库同用户的任务来源原话重建，未使用后来的任务更新。");
    if(latest.reason) line(({no_completed_final_answer:"该轮没有已完成的最终答案，未调用模型。",missing_task_goal:"任务原始目标不可用，保持未知，未调用模型。",lease_expired_result_unknown:"先前等待已过期，没有确认结果；未自动重跑。"})[latest.reason] || "未取得可确认的核对结果。");
    if(expired)line("核对等待已过期，结果未知；可以显式重试，不会自动重跑。");
    for(const r of result?.requirements || []) {
      const item=document.createElement("article");item.className="task-requirement";item.dataset.verdict=r.verdict;
      const label=document.createElement("strong");label.textContent=`${r.id} · ${{met:"模型判满足",violated:"模型判偏离",unknown:"未知"}[r.verdict] || "未知"} · ${r.source_id}`;item.append(label);
      for(const [key,text] of [["任务原话",r.request_quote],["回答节选",r.answer_quote],["模型理由",r.reason]]) {const p=document.createElement("p");p.textContent=`${key}：${text || "无可校验节选"}`;item.append(p);}
      if(r.quote_validation==="failed_unknown") {const p=document.createElement("p");p.textContent="节选校验未通过，该项模型判断未采纳，保留未知；模型理由不能代替有效节选。";item.append(p);}
      box.append(item);
    }
    if(result){line(`核对模型：${result.model || "未记录"}；模型修订未固定。与生成模型相同：${result.same_model_as_generator===true ? "是" : result.same_model_as_generator===false ? "否（仍不证明评审独立可靠）" : "未知"}。`);
      if(result.context_budget?.version==="answer-guard-request-window-v1") {
        const b=result.context_budget;line(`完整请求估算输入 ${b.input_tokens_estimate ?? "未知"}，输出预留 ${b.output_reserved_tokens ?? "未知"}，安全余量 ${b.safety_tokens ?? "未知"}；总计 ${b.estimated_total_tokens ?? "未知"} / ${b.configured_total_tokens ?? "未知"} token。`);
        line(`实报输入 ${b.provider_usage?.prompt_tokens ?? "未知"}，输出 ${b.provider_usage?.completion_tokens ?? "未知"} token；缺用量不填零。`);
        line(`仅正常结束的模型响应可采用；调用状态：${b.reason_code || "未记录"}。本核对的调用单列，不与原问答混算。`);
      }else line("未记录本阶段完整请求预算，保持未知。");
    }
  }
  const button=document.createElement("button");button.type="button";button.className="text-button";
  button.textContent=latest?.status==="running" && !expired ? "正在核对，可刷新对话查看" : result && ["complete","partial"].includes(latest.status) ? "读取已保存核对（相同输入不重复调用）" : "核对本轮任务（额外模型调用）";
  button.disabled=latest?.status==="running" && !expired;button.addEventListener("click",async()=>{button.disabled=true;try{await onReview();}finally{button.disabled=latest?.status==="running" && !expired;}});box.append(button);host.append(box);
}
