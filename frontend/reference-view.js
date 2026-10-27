export const LINK_NAMES = { resolved: "唯一匹配", ambiguous: "同号歧义", missing: "目标缺失", external_unresolved: "外部引用未消解" };
function node(tag, text, className) {
  const value = document.createElement(tag);
  if (text !== undefined) value.textContent = text;
  if (className) value.className = className;
  return value;
}
function sourceCard(host, block, label, onPage) {
  if (!block) return;
  const section = node("section", undefined, "reference-source");
  section.append(node("strong", `${label} · ${block.block_key} · ${block.kind}`));
  section.append(node("pre", block.text || "没有文字。", "segment-text"));
  const recovery = block.caption_recovery;
  if (recovery) {
    section.append(node("p", "表注关联由页内字符范围和唯一相邻表格恢复；请核对表注与表体，尚未处理跨页续表。", "parse-warning"));
    section.append(node("pre", recovery.text, "segment-text"));
    const captionPage = recovery.source_span?.page_number;
    if (captionPage) {
      const captionButton = node("button", `核对恢复表注 · 原页 ${captionPage}`, "text-button");
      captionButton.type = "button";
      captionButton.addEventListener("click", () => onPage(captionPage, [recovery.source_span, ...(block.source_spans || [])]));
      section.append(captionButton);
    }
  }
  const pages = [...new Set((block.source_spans || []).map((span) => span.page_number).filter(Boolean))];
  for (const page of pages) {
    const button = node("button", `查看原页 · ${page}`, "text-button");
    button.type = "button"; button.addEventListener("click", () => onPage(page, block.source_spans)); section.append(button);
  }
  host.append(section);
}
export function renderDocumentReferences(host, response, onPage) {
  host.replaceChildren();
  if (!response.report) host.append(node("p", "尚未建立文内引用索引。可更新索引，无需重新 OCR 或 embedding。", "parse-warning"));
  else {
    const counts = Object.entries(response.report.statuses || {}).map(([key, count]) => `${LINK_NAMES[key] || key} ${count}`).join(" · ");
    host.append(node("p", `共 ${response.report.total} 处编号引用${counts ? ` · ${counts}` : ""}`));
    host.append(node("p", response.report.note, "retrieval-note"));
    if (response.report.truncated) host.append(node("p", "索引达到处理预算或候选上限，尚非全量；请结合原文检查。", "parse-warning"));
    if (response.report.recovered_table_targets?.length) host.append(node("p", `含恢复表注目标：${response.report.recovered_table_targets.join("、")}，可展开条目核对表注及表体原页。`, "parse-warning"));
  }
  if (response.document_status !== "processed") host.append(node("p", "当前文档未就绪，以下可能为历史索引，不参与有效检索。", "parse-warning"));
  if (!response.items.length) host.append(node("p", "没有符合条件的编号引用。"));
  for (const item of response.items) {
    const article = node("article", undefined, "segment-card");
    article.append(node("h3", `${item.source_quote} → ${LINK_NAMES[item.status] || item.status}`));
    sourceCard(article, item.source, "引用来源（节选）", onPage);
    if (item.status === "resolved") {
      for (const target of item.targets) sourceCard(article, target, "目标锚点（节选）", onPage);
      if (item.kind === "figure") article.append(node("p", "这里只定位文字图注，尚未理解图中箭头或嵌套关系。", "parse-warning"));
    } else {
      article.append(node("p", "此引用不会自动展开；同号候选不会按距离或页码擅自选取。", "parse-warning"));
      if (item.status === "ambiguous") for (const target of item.targets) sourceCard(article, target, "歧义候选（最多 20 个）", onPage);
    }
    host.append(article);
  }
}

export function renderReferenceExpansion(host, expansion) {
  if (!expansion) return;
  const details = node("details", undefined, "retrieval-check");
  details.append(node("summary", expansion.enabled ? `文内引用展开：展开前 ${expansion.base_count} 个候选，补充 ${expansion.added_count} 个片段` : "文内引用展开：已关闭"));
  if (expansion.enabled) {
    details.append(node("p", `${expansion.note} 最多 ${expansion.max_hops} 跳，展开仍受数量和上下文预算限制。`, "retrieval-note"));
    if (expansion.truncated) details.append(node("p", "部分目标超过展开预算，当前证据链可能不完整。", "parse-warning"));
    for (const label of expansion.inferred_section_scopes || []) details.append(node("p", `${label}：解析器的标题路径发生变化，已按明确编号区间补充候选；范围仍需核对。`, "parse-warning"));
    for (const label of expansion.recovered_table_targets || []) details.append(node("p", `${label}：表注关联经页内字符范围与唯一相邻表格恢复，仍需核对原页。`, "parse-warning"));
    for (const filename of expansion.unindexed_documents || []) details.append(node("p", `${filename}：引用索引尚未建立或版本已变更，请到“文内引用”更新。`, "parse-warning"));
    for (const item of expansion.unresolved || []) details.append(node("p", `${item.filename} · ${item.label}：${LINK_NAMES[item.status] || item.status}，未自动连接。`, "parse-warning"));
  }
  host.append(details);
}
