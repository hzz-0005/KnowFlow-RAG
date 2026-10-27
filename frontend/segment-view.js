const KINDS = { chunk: "索引切片", text: "正文", table: "表格", picture: "图片区域", section_header: "章节标题", title: "标题", list_item: "列表项", page_header: "页眉", page_footer: "页脚", caption: "图表标题", formula: "公式" };
import { tabularLocations } from "./source-view.js";

function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}

function cellInspector(article, item, onCells) {
  const details = node("details");
  details.append(node("summary", `查看单元格原值 · ${item.tabular_source.cell_count} 格`));
  const content = node("div", undefined, "table-cell-source"); details.append(content);
  let loaded = false; let request = 0;
  async function load(offset) {
    const current = ++request;
    content.textContent = "正在读取原值…";
    try {
      const result = await onCells(item.block_key, offset);
      if (current !== request) return;
      content.replaceChildren();
      content.append(node("p", `${result.sheet_name} · ${result.offset + 1}–${Math.min(result.total, result.offset + result.limit)} / ${result.total} 格；这是保存的原值，公式未重算。`, "retrieval-note"));
      const table = node("table", undefined, "parse-pages");
      const head = node("tr");
      for (const label of ["坐标", "原值 / 格式", "公式", "缓存 / 合并来源"]) head.append(node("th", label));
      table.append(head);
      for (const cell of result.cells) {
        const row = node("tr");
        const raw = cell.raw_value === null ? "[空白，非零]" : JSON.stringify(cell.raw_value);
        row.append(node("td", cell.coordinate), node("td", `${raw}${cell.number_format ? ` · ${cell.number_format}` : ""}`),
          node("td", cell.formula || "—"), node("td", `${cell.formula_cache_status === "missing" ? "未知，未计算" : cell.formula_cache_status === "cached_error" ? `公式缓存错误：${cell.cached_value}` : cell.formula_cache_status === "available_unverified" ? `${JSON.stringify(cell.cached_value)}（未验证）` : "非公式"}${cell.merged_from ? ` · 合并自 ${cell.merged_from}` : ""}${cell.row_hidden ? " · 隐藏行（已纳入）" : ""}`));
        table.append(row);
      }
      content.append(table);
      const previous = node("button", "上一批单元格", "text-button");
      const next = node("button", "下一批单元格", "text-button");
      previous.type = next.type = "button";
      previous.disabled = result.offset === 0; next.disabled = result.offset + result.limit >= result.total;
      previous.addEventListener("click", () => load(Math.max(0, result.offset - result.limit)));
      next.addEventListener("click", () => load(result.offset + result.limit));
      content.append(previous, next);
    } catch (error) { if (current === request) content.textContent = error.message; }
  }
  details.addEventListener("toggle", () => { if (details.open && !loaded) { loaded = true; load(0); } });
  article.append(details);
}

function pdfTableMappings(article, item) {
  const spans = (item.source_spans || []).filter((s) => s.source_mapping === "docling-rendered-row-v1");
  if (!spans.length) return;
  const details = node("details");
  details.append(node("summary", `PDF 表格来源映射 · ${spans.length} 个行片段`));
  details.append(node("p", "区间指向保存的转换文本，采用从 0 开始、右端不含的字符位置；单元格身份来自解析网格。原页仍高亮整表，不代表 OCR、表格结构或语义已验证。", "retrieval-note"));
  const table = node("table", undefined, "parse-pages");
  const head = node("tr");
  for (const label of ["行 / 页", "角色 / 列", "本片段文字", "来源单元格 / 区间"]) head.append(node("th", label));
  table.append(head);
  let count = 0;
  for (const span of spans) for (const cell of span.table_cells || []) {
    if (++count > 100) continue;
    const bounds = cell.chunk_range;
    if (!Array.isArray(bounds) || bounds.length !== 2 || !bounds.every(Number.isInteger) || bounds[0] < 0 || bounds[1] > (item.content || "").length || bounds[0] >= bounds[1]) continue;
    const row = node("tr");
    const source = (cell.source_cells || []).map((s) => `R${s.rows?.join("–")} C${s.columns?.join("–")} (#${s.index})`).join("、");
    row.append(node("td", `行 ${span.row_number} · 页 ${span.page_number}`),
      node("td", `${cell.role === "header" ? "表头" : "值"} · 列 ${cell.column_number}`),
      node("td", item.content.slice(...bounds)),
      node("td", `${source} · 来源 ${(cell.text_range || []).join("–")} / 切片 ${bounds.join("–")}`));
    table.append(row);
  }
  details.append(table);
  if (count > 100) details.append(node("p", `当前展示前 100 项，共 ${count} 项；完整映射保存在切片来源中。`, "retrieval-note"));
  article.append(details);
}

export function renderSegments(host, response, onPage, onCells) {
  host.replaceChildren();
  const stats = response.statistics;
  const name = response.view === "blocks" ? "解析块" : "索引切片";
  host.append(node("p", `共 ${stats.count} 个${name} · 最短 ${stats.min_characters ?? "—"} 字符 · 平均 ${stats.average_characters ?? "—"} 字符 · 最长 ${stats.max_characters ?? "—"} 字符`, "segment-stats"));
  if (response.document_status !== "processed") host.append(node("p", "文档当前未就绪；这里可能保留上次成功的内容，不参与当前检索。", "parse-warning"));
  if (!response.items.length) {
    host.append(node("p", response.view === "blocks" && !stats.count ? "暂无原始解析块。旧索引需要重新解析后才会保存结构。" : "没有符合筛选条件的内容。"));
    return;
  }
  for (const item of response.items) {
    const article = node("article", undefined, "segment-card");
    const heading = node("div", undefined, "segment-heading");
    heading.append(node("strong", `#${item.ordinal + 1} · ${KINDS[item.kind] || item.kind}`), node("span", `${item.characters} 字符`));
    article.append(heading);
    const meta = item.processing_metadata;
    if (meta?.input_tokens != null) article.append(node("p", `${meta.kind} · 完整输入 ${meta.input_tokens} 估算 token${meta.fragmented ? " · 超长元素分片" : ""}`, "retrieval-note"));
    if (item.section) article.append(node("p", item.section, "segment-location"));
    article.append(node("pre", item.content || "此区域没有提取到文字。", "segment-text"));
    const spans = item.source_spans || [];
    const pages = [...new Set([item.page_number, ...spans.map((s) => s.page_number)].filter(Boolean))];
    const actions = node("div", undefined, "segment-actions");
    for (const page of pages) {
      const button = node("button", `查看原页 · ${page}`, "text-button");
      button.type = "button";
      button.addEventListener("click", () => onPage(page, spans)); actions.append(button);
    }
    const cells = tabularLocations(spans);
    for (const location of cells) actions.append(node("span", location, "retrieval-note"));
    if (cells.length) actions.append(node("span", "候选表头与原行一并保留；无 PDF 页码。", "retrieval-note"));
    if (!pages.length && !cells.length) actions.append(node("span", "来源：文档章节（无固定页码）", "retrieval-note"));
    const ids = [...new Set(spans.map((s) => s.block_key).filter(Boolean))];
    if (ids.length || item.block_key) actions.append(node("small", `来源块：${ids.join(", ") || item.block_key}`, "retrieval-note"));
    article.append(actions);
    pdfTableMappings(article, item);
    if (item.tabular_source && onCells) cellInspector(article, item, onCells);
    host.append(article);
  }
}
