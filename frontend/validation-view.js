const STRATEGIES = { vector: "语义检索", keyword: "关键词检索", hybrid: "混合检索", graph: "图谱检索", graph_hybrid: "图谱 + 文本" };
const STAGES = { parse_or_annotation_mismatch: "解析内容或标准引文不匹配", chunking_or_quote_boundary: "分块或引文边界不匹配", retrieval_miss: "未召回", assessment_or_context_filter: "相关性或上下文过滤", refused_with_evidence: "有该证据但最终拒答", evidence_not_cited: "证据未引用", citation_excerpt_shorter_than_gold: "正确来源已引用，节选未全文匹配金标", evidence_chain_complete: "证据链齐全（仍需检查答案）" };
const VERDICTS = { supported: "有依据", unsupported: "缺乏依据", contradicted: "与依据矛盾", not_factual: "非事实陈述", unknown: "未知", satisfied: "已覆盖", missing: "未覆盖" };
const STATUS = { running: "运行中", completed: "运行完成", incomplete: "未完整完成", failed: "运行失败", interrupted: "执行中断", complete: "评审完成", sampled: "仅完成抽样审核", partial: "部分未知", unavailable: "不可用", not_run: "未执行", refused: "已拒答", budget_exceeded: "超出评审预算" };
const GRAPH_STATUS = {processed:"已构建", not_requested:"尚未构建", pending:"等待构建", processing:"构建中", failed:"构建失败"};
const QUALIFIER_KIND = {environment:"环境", version:"版本", condition:"条件", negation:"否定", modality:"情态", time:"时间"};
const SEMANTIC_STATUS = {supported:"模型判支持（引文已校验）", unsupported:"模型判缺乏依据", contradicted:"模型判与原文矛盾", unknown:"审核未知", unreviewed:"旧索引未审核", not_evaluated:"关系语义未验证"};
const fmt = (x) => typeof x === "number" ? x.toFixed(3) : "未知 / 不适用";
function el(tag, text, cls) { const node = document.createElement(tag); if (text != null) node.textContent = text; if (cls) node.className = cls; return node; }
function details(title) { const node = el("details"); node.append(el("summary", title)); return node; }

export function renderGraphEvidence(host, graph, onSource) {
  host.replaceChildren();
  host.append(el("p", `${graph.node_count} 个实体节点 · 当前筛选 ${graph.filtered_edge_count} 条关系`, "segment-stats"), el("p", graph.note, "retrieval-note"));
  if (graph.review_counts) host.append(el("p", `当前实体/文档范围内（不随关系状态筛选改变）可用于图谱检索 ${graph.eligible_edge_count} 条；排除 ${(graph.review_counts.unsupported || 0) + (graph.review_counts.contradicted || 0)} 条；未知 ${graph.review_counts.unknown || 0} 条；旧索引未审核 ${graph.review_counts.unreviewed || 0} 条。未知及未审核关系需重建后再检查。`, "retrieval-note"));
  for (const doc of graph.document_states || []) host.append(el("p", `${doc.filename} · ${GRAPH_STATUS[doc.graph_status] || doc.graph_status}${doc.graph_error ? ` · ${doc.graph_error}` : ""}`, "retrieval-note"));
  if (!graph.edges.length) host.append(el("p", "暂无符合条件的关系。请先构建图谱；空图不能用于证明图谱检索有效。"));
  for (const row of graph.edges) {
    const article = el("article", null, "segment-card");
    article.append(el("strong", `${row.source} → ${row.predicate} → ${row.target}`), el("p", row.description), el("blockquote", row.evidence));
    const qualifiers = row.qualifiers;
    article.append(el("p", qualifiers?.status === "reported" ? `限定引文：${qualifiers.items?.map((q) => `${QUALIFIER_KIND[q.kind] || q.kind}: ${q.text}`).join("；") || "未报告（不代表已证明无条件）"}` : "旧索引未保存结构化限定，不能默认视为无条件事实。", "retrieval-note"));
    article.append(el("p", `${row.filename}${row.page_number ? ` · 第 ${row.page_number} 页` : ""} · 引文存在 ${row.quote_in_source ? "是" : "否"} · 实体名存在 ${row.endpoints_in_source ? "是" : "否"} · ${SEMANTIC_STATUS[row.semantic_correctness] || "关系语义未验证"} · ${row.retrieval_eligible ? "参与图谱检索" : "不参与图谱检索"}`, "retrieval-note"));
    if (row.semantic_review?.reason) article.append(el("p", `审核原因：${row.semantic_review.reason}${row.semantic_review.cache_hit ? " · 复用已有审核" : ""}${row.semantic_review.same_model_as_generator ? " · 同模型审核，仍可能误判" : ""}`, "retrieval-note"));
    const raw = details("查看原始来源切片"); raw.append(el("pre", row.source_text, "segment-text")); article.append(raw);
    if (row.page_number && row.filename.toLowerCase().endsWith(".pdf")) {
      const button = el("button", `查看原页 · ${row.page_number}`, "text-button"); button.type = "button"; button.addEventListener("click", () => onSource(row)); article.append(button);
    }
    host.append(article);
  }
}

export function renderQualityReport(host, report) {
  host.replaceChildren();
  host.append(el("p", `${STATUS[report.status] || report.status} · 数据集 ${report.dataset_version} · top-${report.top_k}`), el("p", report.note, "retrieval-note"));
  host.append(el("p", `指标口径：${report.metrics_version || "旧版（未记录）"}。执行完成表示流程结束，不表示所有质量指标通过。`, "retrieval-note"));
  if (report.metric_replay_of) host.append(el("p", `历史指标重算 · 原运行 ${report.metric_replay_of}；沿用原回答与原模型审核，没有重新问答。`, "retrieval-note"));
  if (report.models) host.append(el("p", `请求模型：普通 ${report.models.generator || "未记录"} · 复杂 ${report.models.complex_generator || report.models.generator || "未记录"} · 审核 ${report.models.judge || "未记录"}。此处是该次运行配置，不代表当前服务或固定模型权重。`, "retrieval-note"));
  if (report.answer_prompt_version) host.append(el("p", `回答提示版本：${report.answer_prompt_version}。提示约束不保证消除无依据推断。`, "retrieval-note"));
  if (report.review_scope) host.append(el("p", report.review_scope.graph_relations ? "本次请求答案及图谱关系评审。" : "本次仅请求答案评审，未请求图谱关系检测。", "retrieval-note"));
  if (report.source_snapshot_unchanged != null) host.append(el("p", `运行期间语料与索引快照${report.source_snapshot_unchanged ? "未变" : "发生变化，结论需复核"}。`, "retrieval-note"));
  if (report.status === "running") host.append(el("p", `${report.phase || "正在执行"} · 已完成 ${report.completed_cases || 0}/${report.total_cases || "—"} 题次`, "retrieval-note"));
  if (report.error || report.comparison_warning) host.append(el("p", report.error || report.comparison_warning, "parse-warning"));
  if (report.baseline_check) {
    const check = report.baseline_check;
    host.append(el("p", `历史回归：${({regressed:"存在退化", incomplete:"存在未知项", no_regression_found:"未发现所检查指标退化", not_comparable:"不可直接比较"})[check.status] || check.status}${check.reason ? ` · ${check.reason}` : ""}`, "retrieval-note"));
    for (const row of check.regressions || []) host.append(el("p", `${row.strategy}/${row.case_id} · ${row.metric}：${fmt(row.before)} → ${fmt(row.after)}`, "parse-warning"));
    for (const row of check.unknown || []) host.append(el("p", row, "parse-warning"));
  }
  if (report.graph_review) {
    const review = report.graph_review;
    host.append(el("p", `关系支持度估算 ${fmt(review.relation_support_estimate)} · 检查 ${review.reviewed_edges}/${review.total_edges} 条 · ${review.sampled ? "仅完成抽样审核，不能视为全量通过" : STATUS[review.status] || review.status}。支持度仅针对已复查关系；这是模型评审，不是图谱准确率。`, "retrieval-note"));
    const inspected = details("查看关系评审明细");
    for (const unit of review.units || []) {
      const edge = report.graph_inspection?.edges?.find((row) => row.id === unit.id);
      inspected.append(el("strong", `${VERDICTS[unit.verdict] || unit.verdict} · ${edge ? `${edge.source} → ${edge.predicate} → ${edge.target}` : unit.text}`));
      inspected.append(el("p", unit.reason || "未提供原因", "retrieval-note"));
      for (const quote of unit.evidence || []) inspected.append(el("blockquote", quote.quote));
    }
    host.append(inspected);
  }
  const variants = Object.entries(report.variants || {});
  if (variants.length > 1) {
    const wrap = el("div", null, "quality-overview"); const table = el("table", null, "parse-pages");
    const head = el("tr"); for (const title of ["同题策略", "证据召回", "MRR", "上下文证据齐全", "答案支持度估算", "要点覆盖估算"]) head.append(el("th", title)); table.append(head);
    for (const [strategy, variant] of variants) {
      const row = el("tr"), m = variant.metrics || {}, d = variant.denominators || {};
      row.append(el("td", STRATEGIES[strategy] || strategy));
      for (const key of [Object.keys(m).find((k) => k.startsWith("recall@")), "mrr", "context_complete_evidence", "faithfulness_estimate", "fact_recall_estimate"]) row.append(el("td", `${fmt(m[key])} (n=${d[key] || 0})`));
      table.append(row);
    }
    wrap.append(table); host.append(wrap);
  }
  for (const [strategy, variant] of variants) {
    host.append(el("h3", STRATEGIES[strategy] || strategy));
    if (variant.performance) host.append(el("p", `在线问答平均 ${fmt(variant.performance.query_mean_ms)} ms · P95 ${fmt(variant.performance.query_p95_ms)} ms。${variant.performance.note}`, "retrieval-note"));
    const metrics = variant.metrics || {}, denominators = variant.denominators || {};
    const table = el("table", null, "parse-pages");
    const head = el("tr"); for (const title of ["指标", "结果", "有效题数"]) head.append(el("th", title)); table.append(head);
    const recall = Object.keys(metrics).find((key) => key.startsWith("recall@"));
    const precision = Object.keys(metrics).find((key) => key.startsWith("precision@"));
    const ndcg = Object.keys(metrics).find((key) => key.startsWith("ndcg@"));
    for (const [key, name] of [[recall, "标准证据召回 Recall"], [precision, "前 k 条标准证据占比 Precision"], ["mrr", "首条相关结果排名 MRR"], [ndcg, "排序 nDCG"], ["base_complete_evidence", "基础召回完整证据率"], ["candidates_evidence_recall", "扩展候选证据召回（包含新增预算）"], ["candidates_complete_evidence", "扩展候选完整证据率"], ["context_complete_evidence", "进入上下文的完整证据率"], ["citation_sources_complete_evidence", "最终引用来源完整覆盖率"], ["citations_complete_evidence", "最终引文金标全文匹配率"], ["refusal_decision_correct", "拒答标志符合标注（非语义判定）"], ["faithfulness_estimate", "答案依据支持度（模型估算）"], ["fact_recall_estimate", "答案要点覆盖（模型估算）"]]) {
      const row = el("tr"); row.append(el("td", name), el("td", fmt(metrics[key])), el("td", String(denominators[key] || 0))); table.append(row);
    }
    host.append(table);
    host.append(el("p", "来源覆盖要求金标已进入上下文，且引用节选来自同一来源；它不证明节选支持答案。金标全文匹配与答案依据审核分别保留。旧报告没有来源覆盖指标时显示未知。", "retrieval-note"));
    for (const group of variant.equivalence_checks || []) host.append(el("p", `同义/中英组 ${group.group}：证据一致 ${group.consistent == null ? "未知" : group.consistent ? "是" : "否"} · 全部证据齐全 ${group.all_complete ? "是" : "否"}`, "retrieval-note"));
    for (const item of variant.cases || []) {
      const box = details(`${item.id} · ${item.question}`);
      if (item.error) { box.append(el("p", item.error, "parse-warning")); host.append(box); continue; }
      box.append(el("p", `最终${item.refused ? "拒答" : "回答"} · ${Math.round(item.latency_ms || 0)} ms`));
      if (item.query_plan) box.append(el("p", `查询改写：${item.query_plan.status} · ${(item.query_plan.search_queries || []).join("；")}。${item.query_plan.note || ""}`, "retrieval-note"));
      if (item.answer_validation) box.append(el("p", `在线回答复核：${item.answer_validation.status} · 保留 ${item.answer_validation.kept_blocks || 0}/${item.answer_validation.total_blocks || 0} 块。${item.answer_validation.note || ""} 草稿及移除原因保存在原始报告中；在线和离线审核可能使用同一个模型。`, "retrieval-note"));
      if (item.answer_validation?.repair) box.append(el("p", `自动纠错：${item.answer_validation.repair.status === "adopted" ? "重写已复核并采用" : "未采用重写"} · 最多一次。首轮复核与纠错草稿保存在原始报告中。`, "retrieval-note"));
      if (item.graph_expansion) {
        const graph = item.graph_expansion;
        box.append(el("p", `候选策略 ${graph.policy}：保留文本 ${graph.base_count} 条，图谱新增 ${graph.added_count}/${graph.max_additions} 条。${graph.note || ""}`, "retrieval-note"));
      }
      const answer = details("查看实际回答"); answer.append(el("pre", item.answer, "segment-text")); box.append(answer);
      for (const fact of item.evidence_checks || []) {
        const card = el("article", null, "segment-card");
        card.append(el("strong", `${fact.filename} · ${STAGES[fact.stage] || fact.stage}`), el("blockquote", fact.contains));
        const ranks = fact.ranks || {};
        card.append(el("p", `基础召回排名：${ranks.base?.join("、") || "未找到"}；展开候选：${ranks.candidates?.join("、") || "未找到"}；进入上下文：${ranks.context?.join("、") || "未进入"}；引用来源：${ranks.citation_sources == null ? "旧报告未记录" : ranks.citation_sources.join("、") || "未引用"}；金标全文匹配引文：${ranks.citations?.join("、") || "未全文匹配"}`, "retrieval-note")); box.append(card);
      }
      const review = item.answer_review || {};
      box.append(el("p", `答案评审：${STATUS[review.status] || review.status || "未执行"}${review.same_model_as_generator ? " · 使用同一模型，存在自评偏差" : ""}`, "retrieval-note"));
      for (const unit of [...(review.units || []), ...(review.facts || [])]) box.append(el("p", `${VERDICTS[unit.verdict] || unit.verdict}：${unit.text} · ${unit.reason || ""}`, unit.verdict === "unknown" ? "parse-warning" : "retrieval-note"));
      host.append(box);
    }
  }
  for (const [strategy, changes] of Object.entries(report.comparisons || {})) {
    host.append(el("h3", `${STRATEGIES[strategy] || strategy} 相对第一种策略`));
    for (const change of changes) host.append(el("p", `${change.id}：${({ improved: "上下文证据改善", regressed: "上下文证据退化", unchanged: "上下文证据无变化", unavailable: "不可比较", not_applicable: "不适用" })[change.status] || change.status} · MRR ${fmt(change.mrr_before)} → ${fmt(change.mrr_after)}`, "retrieval-note"));
  }
}
