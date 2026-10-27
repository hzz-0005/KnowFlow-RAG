import { renderReferenceExpansion } from "./reference-view.js";
import { understandingBudgetLines } from "./understanding-budget-view.js";
import { roundBudgetLines } from "./round-budget-view.js";
import { taskPlanLines } from "./task-plan-view.js";
export const RELEVANCE_NAMES = {
  direct: "直接相关", partial: "部分相关", background: "仅背景", irrelevant: "不相关", unknown: "未核验",
};

function stageBox(box, kind, label) {
  box.classList.add("diagnostic-stage", `diagnostic-${kind}`);
  const badge = document.createElement("span");
  badge.className = "diagnostic-stage-label";
  badge.textContent = label;
  box.querySelector("summary")?.prepend(badge);
}

export function renderRetrievalEvidence(host, response) {
  host.replaceChildren();
  const sourceScope=response.query_plan?.source_scope;
  if(sourceScope?.version==="literal-document-scope-v1" && sourceScope.status!=="unscoped") {
    const box=document.createElement("details"), title=document.createElement("summary");
    const labels={filtered:"已限定文件",unavailable:"指定文件不可用",ambiguous:"文件范围需确认"};
    title.textContent=`检索来源范围：${labels[sourceScope.status] || "状态未知"}`;
    box.append(title);
    const lines=[...(sourceScope.documents||[]).map(d=>d.filename),
      ...(sourceScope.excluded_filenames||[]).map(n=>`排除：${n}`),
      ...(sourceScope.unknown_filenames||[]).map(n=>`当前库未找到：${n}`),
      sourceScope.reason?`范围判定：${sourceScope.reason}`:null,
      "文件过滤在候选排序与数量限制之前执行；未放宽到其他文件。仅限定来源，不保证相关或回答正确。"];
    for(const text of lines.filter(Boolean)){const p=document.createElement("p");p.textContent=text;box.append(p);}
    stageBox(box,"query","来源范围");host.append(box);
  }
  const roundLines=roundBudgetLines(response.round_budget);
  if(roundLines.length) {
    const box=document.createElement("details"),title=document.createElement("summary");
    title.textContent="整轮执行预算：调用与累计预留";box.append(title);
    for(const text of roundLines){const p=document.createElement("p");p.textContent=text;box.append(p);}
    stageBox(box,"budget","整轮预算");host.append(box);
  }
  if (response.context_budget) {
    const budget = response.context_budget;
    const box = document.createElement("details");
    const title = document.createElement("summary");
    const states = {ready:"估算可容纳",blocked:"预算不足",no_evidence_fits:"证据无法整块接纳",
      no_supported_source:"来源检查未通过",no_candidates:"无候选",no_accepted_evidence:"无可用证据",
      output_limit_reached:"生成达到上限"};
    const number = value => Number.isFinite(value) ? String(value) : "未记录";
    title.textContent = `首稿上下文预算：${states[budget.status] || "状态未知"} · 采用 ${number(budget.selected_count)} 块 / 跳过 ${number(budget.skipped_count)} 块`;
    box.append(title);
    const messages = [
      `输入估算 ${number(budget.input_tokens_estimate)} token（固定提示和完整用户条件 ${number(budget.fixed_input_tokens_estimate)}，证据增量 ${number(budget.evidence_tokens_estimate)}）；证据上限 ${number(budget.evidence_cap_tokens)}。`,
      `输出预留 ${number(budget.output_reserved_tokens)} token，安全余量 ${number(budget.safety_tokens)}；估算总占用 ${number(budget.estimated_total_tokens)} / 运行预算 ${number(budget.configured_total_tokens)}。`,
      `输出限额范围：${budget.output_limit_scope === "reasoning_and_answer" ? "思考和正文合计" : "供应商语义尚未核实"}。供应商实报输入 ${number(budget.provider_usage?.prompt_tokens)}、输出 ${number(budget.provider_usage?.completion_tokens)} token；不是本地估算值。`,
      "采用/跳过仅统计首稿装配阶段的可用来源，相关性未采用的其他候选另见候选证据。本地 tokenizer 与供应商分词尚未确认一致；运行预算不是供应商公布的模型窗口。原始条件不截断，证据整块接纳或跳过；仅覆盖首稿生成，后续审核/纠错仍有各自预算。",
    ];
    const route=response.generation_route;
    if (route?.policy_version === "task-route-authoritative-input-v2") {
      const bases={raw_user_intent:"当前及引用的用户原话（含补充）",question_text:"当前完整问题",full_task_fallback:"完整任务（原始意图结构不可用）"};
      const names={extraction_task:"关系抽取",scope_or_negation:"条件/范围/否定词",long_task:"长用户任务",multi_source:"多来源"};
      const signals=Object.entries(route.signals || {}).filter(([,value])=>value === true).map(([key])=>names[key] || "未知规则");
      messages.push(`首稿模型选择：${route.model || "未记录"} · ${route.thinking === "enabled" ? "思考开启" : route.thinking === "disabled" ? "思考关闭" : "供应商默认思考设置"}；判断输入：${bases[route.input_basis] || "未记录"}。`);
      messages.push(`复杂信号：${signals.length ? signals.join("、") : "未命中现有复杂规则"}。辅助提示仍完整发送；这些规则用于分配调用，不是问题难度或答案质量评分。`);
    }
    for (const text of messages) {const p=document.createElement("p");p.textContent=text;box.append(p);}
    const reasons={evidence_cap:"证据额度不足",window_budget:"总预算不足",fixed_input_exhausted:"原始条件及预留已占满",duplicate:"重复内容",empty:"空内容",support_quote_missing:"支持引文不在原块"};
    for (const decision of budget.decisions || []) {
      if (decision.decision !== "skipped") continue;
      const source = (response.retrieval_evidence || []).find(r => r.chunk_id === decision.id);
      const p=document.createElement("p");
      p.textContent=`跳过：${source?.filename || decision.id} · ${reasons[decision.reason] || "原因未知"}。`;
      box.append(p);
    }
    stageBox(box,"budget","首稿预算");
    host.append(box);
  }
  if (response.query_plan && response.query_plan.status !== "disabled" && response.query_plan.status !== "not_requested") {
    const plan = response.query_plan;
    const box = document.createElement("details");
    const title = document.createElement("summary");
    const state = plan.status === "complete"
      ? (plan.reason_code === "no_effective_queries" ? "未增加检索式，使用原问题" : "已完成")
      : (plan.status === "unavailable" ? "不可用，使用原问题" : "状态未知");
    title.textContent = `${plan.planning_mode ? "问题提炼与拆解" : "查询改写"}：${state} · ${Math.round(plan.latency_ms || 0)} ms`;
    box.append(title);
    const reasons = {accepted:"检索式校验通过", no_effective_queries:"返回内容与原问题相同，没有新增检索式",
      local_timeout:"达到本地改写时限", provider_timeout:"供应商请求超时",
      provider_rate_limited:"供应商限流", provider_http_error:"供应商接口返回错误",
      provider_connection_error:"无法连接供应商", provider_error:"供应商响应处理异常",
      client_configuration_error:"改写客户端配置不可用", output_limit:"输出达到上限，未采用截断内容",
      content_filtered:"供应商过滤了输出", finish_reason_missing:"未记录正常结束标记",
      unexpected_finish_reason:"未正常结束", empty_choices:"未返回候选响应", empty_content:"响应正文为空",
      invalid_json:"正文不是合法 JSON", invalid_schema:"JSON 顶层结构不符",
      invalid_query_count:"检索式数量或列表格式不符", invalid_query_text:"检索式类型或长度不符",
      numeric_constraints_changed:"改写增加或遗漏了原问题中的数字，整组未采用",
      request_budget_exceeded:"完整请求超预算，未调用改写模型，保留原查询",
      round_budget_exhausted:"整轮额度不足，未派发改写请求",
      invalid_task_plan_json:"任务计划不是合法JSON，使用原问题",
      invalid_task_plan_schema:"任务计划结构或额外字段不符，使用原问题",
      invalid_task_goal:"目标字段无效",invalid_task_constraints:"条件结构无效",
      invalid_constraint_source:"条件出处无效",constraint_quote_not_unique_exact:"条件无法在原话中唯一逐字定位",
      invalid_task_steps:"步骤结构无效",invalid_task_dependency:"依赖无效或存在前向/循环引用",
      invalid_step_sources:"步骤引用的原话无效",task_constraints_not_linked:"条件关联缺失",
      task_plan_numeric_changed:"子问题增加了所引用原话没有的数字",
      task_plan_numeric_scope_conflict:"子问题借用了其他局部条件的数字，整组未采用",
      too_many_independent_lookups:"独立查询超过上限",task_hint_too_long:"带条件检索线索超长",
      unexpected_error:"未分类的内部异常，原因尚未确认"};
    const phases={configuration:"配置",budget:"请求预算",request:"接口请求",response:"响应完整性",validation:"内容校验",complete:"完成"};
    if (plan.trace_contract === "query-rewrite-diagnostics-v1") {
      const number=value => Number.isFinite(value) ? String(value) : "未记录";
      const rows=[`阶段：${phases[plan.phase] || "未知"}；原因：${reasons[plan.reason_code] || "未知原因"}（${plan.reason_code || "unknown"}）。`,
        `已发起请求：${plan.request_attempted ? "是" : "否"}；已收到响应：${plan.response_received ? "是" : "否"}；仅用原问题：${plan.original_only ? "是" : "否"}；失败回退：${plan.fallback_used ? "是" : "否"}。`,
        `供应商实报输入 ${number(plan.provider_usage?.prompt_tokens)}、输出 ${number(plan.provider_usage?.completion_tokens)} token${plan.http_status != null ? `；HTTP ${plan.http_status}` : ""}。未返回用量时不记作零。`,
        "这些记录只说明改写执行与机械校验状态，不证明同义、否定或范围在语义上完全保真。"];
      for (const text of rows) {const p=document.createElement("p");p.textContent=text;box.append(p);}
      if (plan.protected_numeric_token_count != null && !plan.planning_mode) {
        const p=document.createElement("p");
        p.textContent=`改写输入显式保护 ${number(plan.protected_numeric_token_count)} 个不同数字字面值；仍须逐条通过原有数字集合校验，未自动补写遗漏值。`;
        box.append(p);
      }
      if (plan.validation_failure) {
        const v=plan.validation_failure;const p=document.createElement("p");
        p.textContent=plan.planning_mode
          ? `步骤 ${typeof v.step_id === "string" ? v.step_id : "未知"}：声明 ${number(v.source_ref_count)} 个用户输入引用，未知引用 ${number(v.unknown_source_ref_count)} 个；存在依赖：${v.has_dependency === true ? "是" : v.has_dependency === false ? "否" : "未知"}。未保存无效引用的原值。`
          : `第 ${number(v.query_index)} 条检索式：遗漏 ${number(v.missing_numeric_count)} 个、新增 ${number(v.added_numeric_count)} 个不同数字字面值。`;
        box.append(p);
      }
    } else if (plan.status === "unavailable") {
      const p=document.createElement("p");p.textContent="历史记录未保存具体失败原因，不能从耗时或当前重跑结果反推。";box.append(p);
    }
    for(const text of taskPlanLines(plan.task_plan)) {
      const p=document.createElement("p");p.textContent=text;box.append(p);
    }
    for (const text of [...(plan.search_queries || []), plan.note].filter(Boolean)) {
      const row = document.createElement("p"); row.textContent = text; box.append(row);
    }
    for (const text of understandingBudgetLines(plan.request_budget)) {
      const p=document.createElement("p");p.textContent=text;box.append(p);
    }
    stageBox(box, "query", "查询理解");
    host.append(box);
  }
  if (response.answer_validation && response.answer_validation.status !== "disabled") {
    const review = response.answer_validation;
    const box = document.createElement("details");
    const title = document.createElement("summary");
    const states = {completed:"完成", disabled:"未启用", complete:"复核完成", filtered:"部分块未保留", rejected:"未通过", unavailable:"复核不可用", budget_exceeded:"超出复核预算"};
    const citationOnly=review.version === "document-summary-citation-blocks-v1";
    const total=review.total_blocks ?? (citationOnly && Number.isFinite(review.kept_blocks) && Number.isFinite(review.dropped_blocks) ? review.kept_blocks+review.dropped_blocks : "未知");
    title.textContent = citationOnly
      ? `总结引用标记检查：保留 ${review.kept_blocks ?? "未知"}/${total} 块 · 未做语义复核`
      : `回答依据复核：${states[review.status] || review.status} · 保留 ${review.kept_blocks ?? "未知"}/${total} 块 · ${Math.round(review.latency_ms || 0)} ms`;
    box.append(title);
    if (review.timing_contract === "answer-guard-stages-v1") {
      const trigger=review.repair_trigger || review.initial_review?.repair_trigger;
      if (trigger?.version === "evidence-driven-repair-v1") {
        const reasons={nonfactual_prose_only:"已有可返回的已核验事实，仅省略模型判为非事实的独立叙述块，未额外生成纠错或再次审核",
          repair_disabled:"自动纠错未启用",no_filtered_or_rejected_answer:"本轮无需触发纠错",
          filtered_or_rejected_requires_attention:"存在需要处理的过滤或拒绝，按原策略最多纠错一次并再次审核"};
        const p=document.createElement("p");p.textContent=`纠错触发判断：${reasons[trigger.reason] || "原因未记录"}。这是执行策略，不证明模型语义分类正确。`;box.append(p);
      }
      const names = {initial_review:"首轮审核", repair_generation:"纠错生成", repair_review:"再次审核"};
      for (const stage of review.stages || []) {
        const row = document.createElement("p");
        row.textContent = `${names[stage.stage] || stage.stage}：${states[stage.status] || stage.status} · ${Math.round(stage.latency_ms || 0)} ms${stage.total_units != null ? ` · ${stage.total_units} 个单元 / 未知 ${stage.unknown_units ?? "未记录"}` : ""}${stage.timing ? ` · 复核调用 ${Math.round(stage.timing.review_ms || 0)} ms / 本地校验 ${Math.round(stage.timing.local_validation_ms || 0)} ms` : ""}`;
        box.append(row);
        const budget=stage.context_budget;
        if (budget?.version === "answer-guard-request-window-v1") {
          const n=value=>Number.isFinite(value) ? String(value) : "未记录";
          const yn=value=>value === true ? "是" : value === false ? "否" : "未知";
          const reasons={full_input_exceeds_budget:"完整输入超预算，未发起请求",structural_limit:"超出既有单元/文本限制，未发起请求",
            output_limit_reached:"输出触顶，未采用截断结果",content_filtered:"响应被过滤，未采用结果",
            finish_reason_missing:"缺少正常结束标记，未采用结果",nonfinal_model_output:"响应未正常结束，未采用结果",
            empty_content:"响应正文为空",invalid_response_schema:"审核结构校验失败",repair_text_limit:"纠错文本超出长度限制",
            review_timeout:"审核超时",repair_timeout:"纠错超时",outer_guard_timeout:"达到在线审核外层时限",
            request_cancelled:"请求被取消，完成情况未确认",provider_error:"接口或响应处理失败",review_error:"审核未完成",
            validated_response:"响应完成并经过结构与引文校验",completed_candidate:"纠错候选已生成，仍须再次审核"};
          const p=document.createElement("p");p.textContent=`${names[stage.stage] || "阶段"}预算：完整输入估算 ${n(budget.input_tokens_estimate)} token + 输出预留 ${n(budget.output_reserved_tokens)} + 安全余量 ${n(budget.safety_tokens)} = ${n(budget.estimated_total_tokens)} / 运行目标 ${n(budget.configured_total_tokens)}。发起请求：${yn(budget.request_attempted)}；收到响应：${yn(budget.response_received)}；${reasons[budget.reason_code] || "完成原因未记录"}。供应商实报输入 ${n(budget.provider_usage?.prompt_tokens)}、输出 ${n(budget.provider_usage?.completion_tokens)} token；输出范围：${budget.output_limit_scope === "reasoning_and_answer" ? "思考与正文合计" : "供应商语义未核实"}。`;box.append(p);
        } else {
          const p=document.createElement("p");p.textContent=`${names[stage.stage] || "阶段"}未记录完整请求预算，保持未知。`;box.append(p);
        }
      }
      const note = document.createElement("p");
      note.textContent = "阶段顺序执行，纠错后须再次审核；分项耗时不可与总耗时重复相加，未知不算通过。";
      note.textContent += " 审核输入整体保留或不发起请求，不删待审单元、条件或证据来凑窗口。预算是本地估算，不是供应商真实窗口；未返回用量不填零。";
      box.append(note);
    }
    if (review.repair) {
      const repair = review.repair;
      const language=repair.language_contract || repair.context_budget?.language_contract;
      if(language?.version==="current-user-language-v1") {
        const p=document.createElement("p");
        p.textContent=`纠错语言依据：${language.input_basis==="explicit_current_user_input"?"本轮用户原话（服务器传递）":language.input_basis==="question_text"?"本次完整问题原文":"依据未知"}。显式目标语言/翻译/双语请求优先，否则跟随本轮用户语言；不以审核理由或证据的语言替代。仅为提示约束，未独立验证语言正确；事实仍须再次审核。`;
        box.append(p);
      }
      const row = document.createElement("p");
      row.textContent = `自动纠错（最多一次）：${repair.status === "adopted" ? "重写已通过复核，采用重写结果" : "未采用，保留首轮验证部分或拒答"} · ${Math.round(repair.latency_ms || 0)} ms。${repair.note || ""}`;
      box.append(row);
    }
    const note = document.createElement("p"); note.textContent = review.note || "未通过复核的内容不直接展示为答案。"; box.append(note);
    if (review.citation_binding?.added?.length) {
      const binding = document.createElement("p");
      binding.textContent = `来源绑定：${review.citation_binding.added.length} 个保留块补充了经审核且原文校验通过的来源标记；未使用已拒绝块的引用，不代表独立语义标注。`;
      box.append(binding);
    }
    for (const unit of review.reviewed_blocks || []) {
      const reason = document.createElement("p");
      reason.textContent = `${unit.verdict === "supported" ? "模型判支持" : "未采纳"}：${unit.reason || "未提供原因"}`;
      const text = document.createElement("blockquote"); text.textContent = unit.text;
      box.append(reason, text);
    }
    stageBox(box, "review", "回答核验");
    host.append(box);
  }
  const previous = new Set(host.children);
  renderReferenceExpansion(host, response.reference_expansion);
  for (const box of host.children) if (!previous.has(box) && box.tagName === "DETAILS") stageBox(box, "reference", "证据关联");
  if (response.graph_expansion) {
    const graph = response.graph_expansion;
    const note = document.createElement("p");
    note.className = "retrieval-note";
    note.textContent = `图谱候选补充：保留 ${graph.base_count} 条文本基础候选，去重新增 ${graph.added_count} 条图谱候选（上限 ${graph.max_additions} 条），${graph.shared_count} 条同时被图谱命中。${graph.note || ""}`;
    host.append(note);
  }
  const diagnostics = response.retrieval_diagnostics;
  if (!diagnostics) return;
  if (diagnostics.rerank) {
    const trace = diagnostics.rerank;
    const box = document.createElement("details");
    const title = document.createElement("summary");
    title.textContent = `重排执行：${trace.status === "completed" ? "完成" : "未完成"} · 同池 ${trace.candidate_count} 条 · ${Math.round(trace.latency_ms || 0)} ms`;
    box.append(title);
    for (const text of [`模型：${trace.model || "未知"}；修订固定：${trace.model_revision_pin || "unknown"}；策略回退：${trace.fallback_used ? "是" : "否"}`, trace.note,
      `候选池指纹：${trace.candidate_pool_sha256 || "未知"}`]) {
      const p = document.createElement("p"); p.textContent = text; box.append(p);
    }
    for (const item of trace.ranking || []) {
      const p = document.createElement("p");
      p.textContent = `${item.chunk_id}：原排名 ${item.before_rank} → 重排 ${item.after_rank}${item.after_rank <= trace.requested_top_k ? " · 前 k" : ""}`;
      box.append(p);
    }
    stageBox(box, "ranking", "排序执行");
    host.append(box);
  }
  if (diagnostics.parsing_warnings?.length) {
    const box = document.createElement("details");
    const title = document.createElement("summary");
    title.textContent = `解析与来源提示：${diagnostics.parsing_warnings.length} 项待核对`;
    box.append(title);
    for (const text of diagnostics.parsing_warnings) {
      const warning = document.createElement("p");
      warning.className = "parse-warning retrieval-note";
      warning.textContent = text;
      box.append(warning);
    }
    stageBox(box, "warning", "文档质量");
    host.append(box);
  }
  const evidence = response.retrieval_evidence || [];
  const root = document.createElement("details");
  root.className = "retrieval-check";
  root.open = Boolean(response.refused);
  const summary = document.createElement("summary");
  const counts = {};
  for (const row of evidence) {
    const verdict = row.assessment?.verdict || "unknown";
    counts[verdict] = (counts[verdict] || 0) + 1;
  }
  const labels = Object.entries(RELEVANCE_NAMES).filter(([key]) => counts[key])
    .map(([key, name]) => `${name} ${counts[key]}`).join(" · ");
  summary.textContent = `检索相关性检查：${evidence.length} 个候选片段${labels ? `（${labels}）` : ""}`;
  root.append(summary);
  stageBox(root, "evidence", "候选证据");
  const note = document.createElement("p");
  note.className = "retrieval-note";
  note.textContent = diagnostics.note;
  root.append(note);
  const assessmentBudget=diagnostics.context_budget;
  if (["relevance-window-whole-passages-v1", "relevance-window-source-metadata-v2"].includes(assessmentBudget?.version)) {
    const n=value => Number.isFinite(value) ? String(value) : "未记录";
    const causes={fixed_input_exhausted:"完整问题与提示已占满预算，未调用检测模型",
      no_whole_passage_fits:"没有候选可完整装入，未调用检测模型",output_limit_reached:"输出触顶，未采用截断判断",
      content_filtered:"响应被供应商过滤，未采用判断",finish_reason_missing:"缺少正常结束标记，未采用判断",
      nonfinal_model_output:"响应未正常结束，未采用判断",empty_content:"响应正文为空",
      invalid_response_schema:"响应结构校验失败",client_configuration_error:"客户端配置不可用",
      provider_error:"接口调用或响应处理失败",validated_response:"响应完成并经格式及引文校验"};
    const texts=[`相关性请求预算：装配 ${n(assessmentBudget.selected_count)} 块 / 未装配 ${n(assessmentBudget.skipped_count)} 块；取得可用判断 ${n(assessmentBudget.observed_count)} 个身份，未核验 ${n(assessmentBudget.unknown_count)} 个身份。`,
      `完整输入估算 ${n(assessmentBudget.input_tokens_estimate)} token（提示与用户条件 ${n(assessmentBudget.fixed_input_tokens_estimate)}，证据增量 ${n(assessmentBudget.evidence_tokens_estimate)}）；输出预留 ${n(assessmentBudget.output_reserved_tokens)}，安全余量 ${n(assessmentBudget.safety_tokens)}；合计 ${n(assessmentBudget.estimated_total_tokens)} / 运行目标 ${n(assessmentBudget.configured_total_tokens)}。`,
      `已发起请求：${assessmentBudget.request_attempted ? "是" : "否"}；已收到响应：${assessmentBudget.response_received ? "是" : "否"}；${causes[assessmentBudget.reason_code] || "状态原因未记录"}。`,
      `供应商实报输入 ${n(assessmentBudget.provider_usage?.prompt_tokens)}、输出 ${n(assessmentBudget.provider_usage?.completion_tokens)} token；未返回用量不填零。输出限额范围：${assessmentBudget.output_limit_scope === "reasoning_and_answer" ? "思考与正文合计" : "供应商语义未核实"}。`,
      "候选整块接纳或跳过，不截断半句；未核验不等于不相关。运行目标及本地估算不是供应商真实窗口，不证明语义正确；本面板只覆盖相关性检测阶段。"];
    for (const text of texts) {const p=document.createElement("p");p.textContent=text;root.append(p);}
  } else if (!["empty","disabled"].includes(diagnostics.status)) {
    const p=document.createElement("p");p.textContent="历史记录未保存本阶段完整请求预算与结束原因，保持未知。";root.append(p);
  }
  if (diagnostics.status === "unavailable") {
    const warning = document.createElement("p");
    warning.textContent = "本次检测暂不可用，片段未核验。";
    root.append(warning);
  }
  for (const item of evidence) {
    const row = document.createElement("details");
    row.className = "retrieval-row";
    const top = document.createElement("summary");
    const title = document.createElement("span");
    title.className = "retrieval-source";
    title.textContent = `${item.rank}. ${item.filename}`;
    title.title = item.section || item.filename;
    const badge = document.createElement("span");
    const verdict = Object.hasOwn(RELEVANCE_NAMES, item.assessment?.verdict)
      ? item.assessment.verdict : "unknown";
    badge.className = `relevance-badge relevance-${verdict}`;
    row.dataset.verdict = verdict;
    badge.textContent = RELEVANCE_NAMES[verdict];
    const usage = document.createElement("span");
    usage.className = "retrieval-usage";
    usage.textContent = item.cited_in_answer ? "已引用" : item.included_in_context ? "已提供给模型" : "未采用";
    top.append(title, badge, usage);
    row.append(top);
    const reason = document.createElement("p");
    reason.textContent = item.assessment?.reason || "尚未核验。";
    row.append(reason);
    const metrics = document.createElement("p");
    metrics.className = "retrieval-note";
    const scores = [];
    if (item.distance_gate_fallback) scores.push("空召回恢复：超过距离阈值，需核验");
    if (item.retrieval_stage === "base") scores.push("基础召回");
    if (item.retrieval_stage === "text_expansion") scores.push("英文词干补充候选（基础 k 之外）");
    if (item.retrieval_stage === "graph_expansion") scores.push("图谱补充候选");
    if (item.retrieval_stage === "reference_expansion") scores.push("文内引用补充候选");
    if (Number.isFinite(item.cosine_similarity)) scores.push(`余弦相似度 ${item.cosine_similarity.toFixed(3)}`);
    if (item.vector_rank) scores.push(`向量排名 ${item.vector_rank}`);
    if (item.keyword_rank) scores.push(`关键词排名 ${item.keyword_rank}`);
    if (item.english_keyword_rank) scores.push(`英文词干排名 ${item.english_keyword_rank}`);
    if (item.graph_rank) scores.push(`图谱排名 ${item.graph_rank}`);
    if (item.table_rank) scores.push(`表格行排名 ${item.table_rank}`);
    if (Number.isFinite(item.rerank_score)) scores.push(`重排分数 ${item.rerank_score.toFixed(3)}`);
    if (!scores.length) scores.push("没有可用的原始检索分数");
    metrics.textContent = scores.join(" · ") + "；这些不是正确率。";
    row.append(metrics);
    const location = document.createElement("p");
    location.className = "retrieval-note";
    location.textContent = [item.section, item.page_number ? `第 ${item.page_number} 页` : null].filter(Boolean).join(" · ");
    row.append(location);
    for (const path of item.reference_paths || []) {
      const info = document.createElement("p");
      info.className = "retrieval-note";
      info.textContent = `文内引用 ${path.label} · 第 ${path.hop} 跳 · 来源片段 ${path.source_chunk_id} · 原文“${path.source_quote}”`;
      row.append(info);
    }
    const quote = document.createElement("blockquote");
    quote.textContent = item.quote;
    row.append(quote);
    root.append(row);
  }
  host.append(root);
}
