function node(tag, text, className) {
  const value = document.createElement(tag);
  if (text !== undefined) value.textContent = text;
  if (className) value.className = className;
  return value;
}

export function renderOcrReview(host, review, onSource) {
  host.replaceChildren();
  const names = { not_started: "尚未运行单页 OCR 对照", running: "单页 OCR 正在运行，可关闭页面后回来查看", succeeded: "对照完成，待人工核对", failed: "对照失败，现有索引未修改" };
  host.append(node("p", names[review.status] || "对照状态未知", "parse-status"));
  if (review.created_at) host.append(node("p", `最近任务：${new Date(review.created_at).toLocaleString("zh-CN")} · 原页 ${review.page_number}`, "retrieval-note"));
  if (!review.parser_available) host.append(node("p", "未启用 Docling 结构化解析服务，暂时无法运行 OCR。", "parse-warning"));
  else if (!review.can_start && review.document_status === "processed") host.append(node("p", "此 PDF 需要先重新解析并保存结构化来源，才能进行对照。", "parse-warning"));
  if (review.baseline_changed) host.append(node("p", "当前入库来源已变化，以下为历史基线；请重新运行后再比较。", "parse-warning"));
  if (review.document_status !== "processed") host.append(node("p", "当前文档未就绪，请先完成文档处理。", "parse-warning"));
  if (review.error_message) host.append(node("p", review.error_message, "parse-warning"));
  const source = node("button", `查看原页 · ${review.page_number}`, "text-button");
  source.type = "button"; source.addEventListener("click", () => onSource(review.page_number)); host.append(source);
  const { baseline, candidate, comparison, candidate_report: report } = review.result || {};
  if (!baseline) return;
  if (baseline.partitioned_page_blocks?.length) host.append(node("p", `当前页有 ${baseline.partitioned_page_blocks.length} 个跨页文字块已按完整字符范围分离；对照正文仅保留本页部分。`, "retrieval-note"));
  if (comparison) {
    const retention = comparison.baseline_token_retention;
    host.append(node("p", `当前正文词元保留比例：${retention == null ? "无法计算" : `${(retention * 100).toFixed(1)}%`}；它只度量文字重合，不是 OCR 准确率。`, "retrieval-note"));
    for (const warning of comparison.warnings || []) host.append(node("p", warning, "parse-warning"));
    if (comparison.note) host.append(node("p", comparison.note, "retrieval-note"));
  }
  const table = node("table", undefined, "parse-pages");
  const headings = node("tr");
  for (const label of ["排查项", "当前入库正文", "强制 OCR 候选"]) headings.append(node("th", label));
  const thead = node("thead"); thead.append(headings); table.append(thead);
  const tbody = node("tbody");
  for (const [label, field] of [["字符数", "characters"], ["替换字符", "replacement_characters"], ["正文块", "block_count"], ["表格块", "table_count"]]) {
    const row = node("tr"); row.append(node("th", label), node("td", baseline[field] ?? "未知"), node("td", candidate?.[field] ?? "待生成")); tbody.append(row);
  }
  table.append(tbody); host.append(table);
  if (comparison) {
    for (const [label, items] of [["候选中未保留的数字", comparison.missing_numbers], ["候选新增的数字", comparison.added_numbers]]) {
      host.append(node("p", `${label}：${items?.length ? items.map((item) => `${item.value}（${item.count} 次）`).join("、") : "无"}`, "ocr-numbers"));
    }
    if (comparison.number_details_truncated) host.append(node("p", "数字变化较多，各列最多展示 50 种。", "retrieval-note"));
  }
  const columns = node("div", undefined, "ocr-columns");
  for (const [title, snapshot] of [["当前入库正文", baseline], ["强制 OCR 候选 · 尚未入库", candidate]]) {
    const section = node("section"); section.append(node("h3", title));
    section.append(node("pre", snapshot ? snapshot.text || "此页没有正文文字。" : "运行完成后展示候选。", "ocr-text")); columns.append(section);
  }
  host.append(columns);
  if (report?.warnings?.length) {
    const details = node("details"); details.append(node("summary", "候选解析告警"));
    for (const warning of report.warnings) details.append(node("p", warning, "parse-warning")); host.append(details);
  }
}
