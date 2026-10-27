import { renderDiagnostics } from "./diagnostics-view.js";

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
const NAMES = { inspected: "已检查", unavailable: "来源不可用", not_applicable: "不适用", deferred: "暂未检查", error: "本次检查中断" };
const REASONS = { document_missing: "文档已删除或不可用", rejected_no_document: "拒收文件没有索引",
  document_in_progress: "文档仍在处理中", batch_budget_exhausted: "达到本批检查预算", index_changed: "检查时索引发生变化，请重查" };

export function renderBatchDiagnostics(host, data, onInspect) {
  host.replaceChildren();
  host.append(el("p", data.note, "retrieval-note"));
  const exportButton = el("button", "导出本次诊断", "button button-small button-outline");
  exportButton.type = "button";
  exportButton.addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json;charset=utf-8" }));
    const link = el("a"); link.href = url;
    link.download = `knowflow-batch-diagnostics-${String(data.batch_id || "snapshot").replace(/[^a-zA-Z0-9-]/g, "")}.json`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  host.append(exportButton, el("p", "本次结果按需计算，未在服务端保存独立历史；需要对照时请导出。", "retrieval-note"));
  const summary = data.summary;
  host.append(el("p", `输入 ${summary.n_inputs} 个 · 已检查输入 ${summary.inspection_status_counts.inspected} 个 · 实际检查唯一文档 ${summary.n_unique_inspected_documents} 份`, "parse-status"));
  if (data.changed_during_inspection) host.append(el("p", "检查期间批任务或文档发生变化；下列是读取时的记录，请重新检查后再判断当前状态。", "parse-warning"));
  host.append(el("p", `本次读取 ${(data.budget.read_bytes / 1024).toFixed(1)} KiB；达到预算会保留“暂未检查”行。不是质量通过率。`, "retrieval-note"));
  const wrapper = el("div", undefined, "table-cell-source");
  const table = el("table", undefined, "parse-pages");
  const head = el("tr");
  for (const label of ["输入文件", "检查状态", "发现与限制"]) head.append(el("th", label));
  table.append(head);
  for (const item of data.items) {
    const row = el("tr");
    const doc = data.documents[item.document_id];
    const unknowns = doc?.findings.filter(f => f.severity === "unknown").length || 0;
    const issues = doc?.findings.filter(f => ["error", "warning"].includes(f.severity)).length || 0;
    const partial = doc && (!doc.scope.complete || !doc.scope.content_complete || !doc.scope.sources_complete);
    const note = doc ? `${issues} 类异常/需核对 · ${unknowns} 类未知${partial ? " · 部分检查" : ""}${doc.stale ? " · 文档已变化" : ""}${item.reused ? " · 复用本次同一来源检查" : ""}；展开下方文档详情` : REASONS[item.reason] || item.reason;
    row.append(el("td", item.filename), el("td", NAMES[item.inspection_status] || item.inspection_status), el("td", note));
    table.append(row);
  }
  wrapper.append(table); host.append(wrapper);
  host.append(el("p", "下列详情按唯一文档展示，重复输入不会重复计入异常文档数。", "retrieval-note"));
  for (const [id, doc] of Object.entries(data.documents)) {
    const details = el("details", undefined, "diagnostic-finding");
    details.append(el("summary", `${doc.filename}${doc.stale ? "（读取后已变化）" : ""}`));
    const body = el("div"); details.append(body);
    let rendered = false;
    details.addEventListener("toggle", () => {
      if (!details.open || rendered) return;
      rendered = true;
      renderDiagnostics(body, doc, example => onInspect(id, example));
    });
    host.append(details);
  }
  host.append(el("p", `检查时间：${new Date(data.started_at).toLocaleString("zh-CN")} 至 ${new Date(data.completed_at).toLocaleString("zh-CN")} · ${data.schema_version}`, "diagnostic-fingerprint"));
}
