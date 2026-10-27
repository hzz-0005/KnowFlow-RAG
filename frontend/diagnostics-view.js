function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

const SEVERITIES = { error: "定位异常", warning: "需核对", unknown: "无法判断", info: "提示" };
const CONSISTENCY_LABELS = { match: "一致", normalized_match: "规范化一致", mismatch: "不一致", not_comparable: "无法比较" };
const CONSISTENCY_BADGE = { match: "b-match", normalized_match: "b-normalized", mismatch: "b-mismatch", not_comparable: "b-unknown" };
const REASON_LABELS = {
  missing_chunk_range:"缺少切片区间", missing_text_range:"缺少来源字符区间",
  invalid_ranges:"区间无效", source_text_unavailable:"来源正文未读到",
  source_text_truncated:"来源正文读取被截断", excerpt_text_truncated:"切片正文读取被截断",
  ambiguous_character_mapping:"来源与切片范围不能确定一一对应",
  comparison_budget_exhausted:"达到文字比对预算",
};

function renderConsistency(host, consistency) {
  const section = el("section", undefined, "consistency-block");
  section.append(el("h3", "来源文字与切片节选一致性"));
  if (consistency.status === "not_run") {
    section.append(el("p", consistency.note, "retrieval-note"));
    host.append(section);
    return;
  }
  const spans = consistency.spans;
  if (consistency.status === "unknown") section.append(el("p", "本次没有可判定的文字比对结果，保留未知。", "parse-warning"));
  section.append(el(
    "p",
    `契约 ${consistency.contract} · 可比对 ${spans.compared} 条（一致 ${spans.match} · 规范化一致 ${spans.normalized_match} · 不一致 ${spans.mismatch}）· 无法比较 ${spans.not_comparable} 条`,
    "parse-status",
  ));
  const reasons = Object.entries(consistency.reasons || {});
  if (reasons.length) {
    section.append(el("p", `无法比较的原因：${reasons.map(([key, count]) => `${REASON_LABELS[key] || key}（${key}） ×${count}`).join(" · ")}`, "retrieval-note"));
  }
  if (consistency.coverage) section.append(el("p", `已读切片 ${consistency.coverage.chunks_inspected} 个 · 无来源项 ${consistency.coverage.chunks_without_span_objects} 个（不计入来源项分母）· 来源项 ${consistency.coverage.span_objects_inspected} 条`, "retrieval-note"));
  if (consistency.work) section.append(el("p", `文字比对用量 ${consistency.work.characters_inspected} / ${consistency.work.character_budget} 字符；达到预算后保留未知。`, "retrieval-note"));
  for (const example of consistency.examples || []) {
    const pair = el("div", undefined, `consistency-pair${example.status === "mismatch" ? " is-mismatch" : ""}`);
    const head = el("div", undefined, "consistency-head");
    head.append(el("span", CONSISTENCY_LABELS[example.status] || example.status, `consistency-badge ${CONSISTENCY_BADGE[example.status] || ""}`));
    const labels = [];
    if (Number.isInteger(example.ordinal)) labels.push(`切片 #${example.ordinal + 1}`);
    if (example.block_key) labels.push(`来源块 ${example.block_key}`);
    if (Number.isInteger(example.page_number)) labels.push(`第 ${example.page_number} 页`);
    if (example.reason) labels.push(`不可比较原因：${REASON_LABELS[example.reason] || example.reason}`);
    if (example.normalizations?.length) labels.push(`规范化：${example.normalizations.join(" + ")}`);
    if (example.chunk_range) labels.push(`节选区间 ${example.chunk_range[0]}–${example.chunk_range[1]}`);
    if (example.text_range) labels.push(`来源区间 ${example.text_range[0]}–${example.text_range[1]}`);
    if (example.difference) labels.push(`首个差异：来源第 ${example.difference.source_offset + 1} 字符 / 节选第 ${example.difference.excerpt_offset + 1} 字符（按 Unicode 码点计）`);
    head.append(el("span", labels.join(" · ")));
    pair.append(head);
    const grid = el("div", undefined, "consistency-grid");
    for (const [title, value, truncated] of [
      ["来源文字", example.source, example.source_truncated],
      ["切片节选", example.excerpt, example.excerpt_truncated],
    ]) {
      const column = el("div");
      column.append(el("h4", truncated ? `${title}（截断展示）` : title));
      column.append(el("p", value ?? "（本次未读到文字，无法对照）", "consistency-text"));
      grid.append(column);
    }
    pair.append(grid);
    section.append(pair);
  }
  section.append(el("p", consistency.note, "retrieval-note"));
  host.append(section);
}

export function renderDiagnostics(host, data, onInspect) {
  host.replaceChildren();
  host.append(el("p", data.note, "retrieval-note"));
  const scope = data.scope;
  host.append(el("p", `检查范围：解析块 ${scope.inspected.blocks} / ${scope.total.blocks} · 切片 ${scope.inspected.chunks} / ${scope.total.chunks}${scope.complete && scope.content_complete && scope.sources_complete !== false ? "（已读全量记录）" : "（部分检查）"}`, "parse-status"));
  if (scope.source_texts_complete === false) host.append(el("p", "来源正文读取不完整：未读到或超出预算的区间保留未知；记录数量读全不等于正文比对完整。", "parse-warning"));
  const tokens = data.observations.token_budget;
  host.append(el("p", `历史 token 预算重算：可核对 ${tokens.observed} 个 · 未知 ${tokens.unknown} 个（均为已读切片）；最高 ${tokens.max_input_tokens ?? "未知"} / ${tokens.saved_budget ?? "未知"} 估算 token（已扣预留量）`));
  const pages = data.observations.page_coverage;
  host.append(el("p", pages.status === "not_applicable"
    ? "此格式不适用 PDF 页覆盖；表格按工作表与单元格定位。"
    : pages.status === "observed"
      ? `有切片来源的页：${pages.pages_with_chunk_source} / ${pages.expected_pages}；不代表文字或语义完整覆盖。`
      : "历史页覆盖：未知，缺少页数或未读全量切片/来源。", "retrieval-note"));
  if (data.observations.source_consistency) renderConsistency(host, data.observations.source_consistency);
  if (!data.findings.length) host.append(el("p", "已检查范围内未发现这些机械异常；语义完整性仍需评测。"));
  for (const finding of data.findings) {
    const section = el("section", undefined, `diagnostic-finding diagnostic-${finding.severity}`);
    section.append(el("h3", `${SEVERITIES[finding.severity] || finding.severity} · ${finding.count} 处`), el("p", finding.message));
    if (finding.examples?.length) {
      const details = el("details");
      details.append(el("summary", `查看位置（展示 ${finding.examples.length} / ${finding.count}）`));
      for (const example of finding.examples) {
        const row = el("p", undefined, "diagnostic-location");
        const labels = [];
        if (Number.isInteger(example.ordinal)) labels.push(`切片 #${example.ordinal + 1}`);
        if (example.block_key) labels.push(`来源块 ${example.block_key}`);
        if (Number.isInteger(example.page_number)) labels.push(`第 ${example.page_number} 页`);
        if (example.sheet_name) labels.push(`${example.sheet_name} · ${example.cell_range || "范围未知"}`);
        if (Number.isInteger(example.input_tokens)) labels.push(`${example.input_tokens} / ${example.budget} token`);
        row.append(el("span", labels.join(" · ")));
        if (example.chunk_id || example.block_key) {
          const button = el("button", example.chunk_id ? "查看切片列表" : "查看解析块列表", "text-button");
          button.type = "button";
          button.addEventListener("click", () => onInspect(example));
          row.append(button);
        }
        details.append(row);
      }
      section.append(details);
    }
    host.append(section);
  }
  host.append(el("p", `规则版本：${data.schema_version} · 已读数据指纹：${data.snapshot_sha256}`, "diagnostic-fingerprint"));
}
