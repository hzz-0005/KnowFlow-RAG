function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

export function tabularLocations(spans) {
  return [...new Set((spans || []).filter((s) => s.source_type === "spreadsheet")
    .map((s) => `${s.sheet_name} · ${s.cell_range || `行 ${s.row_start}–${s.row_end}`}`))];
}

export function renderParsingReport(host, doc, onPage) {
  host.replaceChildren();
  const report = doc.parsing_report || {};
  if (report.comparison_copy?.reused_parser_and_vectors) host.append(element("p", "来源定位对照副本：复用已保存解析结果与向量，未重新 OCR 或调用模型。", "retrieval-note"));
  const names = { parsed: "结构解析完成", needs_review: "解析需核对", failed: "解析失败" };
  if (["pending", "processing"].includes(doc.status)) {
    host.append(element("p", report.parser ? "文档正在处理；以下为上一次的解析报告，本次结果尚未生成。" : "文档正在处理，解析报告将在处理完成后显示。", "retrieval-note"));
    if (!report.parser) return;
  }
  host.append(element("p", names[report.status] || "尚无解析报告，旧索引需重新解析后才能查看。", "parse-status"));
  if (!report.parser) return;
  if (report.parser === "tabular") {
    host.append(element("p", `${report.source_format?.toUpperCase()} · ${report.sheet_count} 个工作表 · ${report.table_count} 个数据区域；按单元格定位，无固定页码。`));
    for (const sheet of report.sheets || []) host.append(element("p", `${sheet.sheet_name} · ${sheet.nonempty_rows} 个非空行 · ${sheet.column_count} 列${sheet.sheet_state !== "visible" ? " · 隐藏工作表（已纳入索引）" : ""}${sheet.merged_ranges?.length ? ` · 合并区域 ${sheet.merged_ranges.join("、")}` : ""}`, "retrieval-note"));
    host.append(element("p", `公式缓存：缺失 ${report.tabular?.formula_cells_missing_cache || 0} 个 · 已保存但未验证 ${report.tabular?.formula_cells_cached || 0} 个 · 错误 ${report.tabular?.formula_cells_cached_error || 0} 个；本系统没有执行公式。`, "retrieval-note"));
    if (report.tabular?.encoding) host.append(element("p", `CSV 编码：${report.tabular.encoding}；分隔符：${JSON.stringify(report.tabular.delimiter)}`, "retrieval-note"));
  } else {
    host.append(element("p", `解析器：${report.parser} · ${report.page_count ?? 0} 页 · ${report.block_count ?? 0} 个块 · ${report.table_count ?? 0} 个表格 · ${report.picture_count ?? 0} 个图片区域`));
    host.append(element("p", report.comparison_copy?.reused_parser_and_vectors
      ? (report.ocr_requested ? "历史解析记录曾请求 OCR；本次对照没有重新 OCR，识别结果仍需与原页核对。" : "历史解析记录未请求 OCR；本次对照也未运行 OCR。")
      : (report.ocr_requested ? "已请求 OCR；识别结果仍需与原页核对。" : "本次未运行 OCR。")));
  }
  if (report.chunking?.strategy === "structure_tokens") {
    const chunking = report.chunking;
    host.append(element("p", `自动分块：${chunking.chunk_count} 个 · ${chunking.cross_page_chunks} 个跨页块 · 完整输入最高 ${chunking.max_input_tokens} / ${chunking.max_tokens} 估算 token（预留 ${chunking.token_reserve}）`, "retrieval-note"));
    host.append(element("p", chunking.note, "retrieval-note"));
    if (chunking.source_mapping_revision) host.append(element("p", "来源定位已按片段内相对区间保存；无字符映射的表格仍保留未知，字符对应不代表阅读顺序或语义已验证。", "retrieval-note"));
    if (chunking.source_mapping_revision === "piece-and-table-rows-v2") host.append(element("p", "PDF 表格先校验整表转换文本及行顺序，再保存行区间与单元格身份；可在索引切片展开查看。跨页或结构不一致时不推测映射，原页高亮仍是整表区域。", "retrieval-note"));
  }
  if (report.parse_artifact?.available) host.append(element("p", report.comparison_copy?.reused_parser_and_vectors
    ? "本次对照沿用已保存的解析产物，仅重算来源区间。"
    : (report.parse_artifact.reused ? "本次复用了已校验解析产物，没有重新 OCR。" : "已保存解析产物，修改分块策略可直接重新分块。"), "retrieval-note"));
  if (report.graph_index) host.append(element("p", `图谱：${report.graph_index.source_chunks} 个来源切片 · ${report.graph_index.entities} 个实体 · ${report.graph_index.relations} 条关系 · 抽取阶段排除/去重 ${report.graph_index.rejected_batch_relations} 条候选。${report.graph_index.note}`, "retrieval-note"));
  if (report.table_caption_recovery) {
    const captions = report.table_caption_recovery;
    host.append(element("p", `页内表注恢复：${captions.recovered} 个关联 · ${captions.candidates} 个候选；恢复依据与原页可在“文内引用”核对。`, "retrieval-note"));
    if (captions.truncated) host.append(element("p", "表注恢复达到处理预算，部分页面未检查；结果并非全量。", "parse-warning"));
  }
  if (report.reference_index) {
    const refs = report.reference_index;
    host.append(element("p", `编号引用 ${refs.total} 处 · 唯一匹配 ${refs.statuses?.resolved || 0} 处 · 未消解 ${refs.total - (refs.statuses?.resolved || 0)} 处；这里只记录结构链接。`, "retrieval-note"));
  }
  for (const warning of report.warnings || []) host.append(element("p", warning, "parse-warning"));
  if (report.error) host.append(element("p", report.error, "parse-warning"));
  if (report.indexing_failed) host.append(element("p", "解析已返回，但索引构建失败，本次内容尚不可检索。", "parse-warning"));
  host.append(element("p", report.note || "解析完成不代表内容正确率，不能据此保证信息没有遗漏。", "retrieval-note"));
  if (!(report.pages || []).length) return;
  const table = element("table", undefined, "parse-pages");
  const head = element("thead");
  const titles = element("tr");
  for (const title of ["原页", "原文字层字符", "索引块字符", "核对"]) titles.append(element("th", title));
  head.append(titles); table.append(head);
  const body = element("tbody");
  for (const page of report.pages) {
    const row = element("tr");
    row.append(element("td", `第 ${page.page_number} 页`), element("td", page.native_characters ?? "未知"), element("td", page.indexed_characters ?? "未知"));
    const action = element("td");
    const button = element("button", "查看原页", "text-button");
    button.type = "button";
    button.addEventListener("click", () => onPage(page.page_number));
    action.append(button); row.append(action); body.append(row);
  }
  table.append(body); host.append(table);
}

export function renderSourcePage(host, url, pageNumber, spans) {
  host.replaceChildren();
  const matching = (spans || []).filter((span) => span.page_number === pageNumber && span.bbox);
  host.append(element("p", matching.length ? "高亮为引用所属内容块的区域，可能包含表格或完整段落；请结合引文核对。" : "此引用暂无区域坐标，当前显示原页供人工核对。", "retrieval-note"));
  const wrapper = element("div", undefined, "source-page");
  const img = element("img");
  img.src = url; img.alt = `文档原页，第 ${pageNumber} 页`;
  wrapper.append(img);
  for (const span of matching) {
    const { left, top, right, bottom } = span.bbox;
    if (![left, top, right, bottom].every((v) => Number.isFinite(v) && v >= 0 && v <= 1) || right <= left || bottom <= top) continue;
    const box = element("span", undefined, "source-highlight");
    box.style.left = `${left * 100}%`; box.style.top = `${top * 100}%`;
    box.style.width = `${(right - left) * 100}%`; box.style.height = `${(bottom - top) * 100}%`;
    box.setAttribute("aria-hidden", "true"); wrapper.append(box);
  }
  host.append(wrapper);
}
