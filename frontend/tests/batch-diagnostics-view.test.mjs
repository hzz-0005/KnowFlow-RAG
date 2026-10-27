import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { renderBatchDiagnostics } from "../batch-diagnostics-view.js";

function render(overrides = {}) {
  globalThis.document = new JSDOM("<main></main>").window.document;
  const host = document.querySelector("main");
  renderBatchDiagnostics(host, {
    note: "机械检查，不是质量通过", summary: { n_inputs: 3, n_unique_inspected_documents: 1, inspection_status_counts: { inspected: 2 } },
    budget: { read_bytes: 1000 }, started_at: "2026-10-23T00:00:00Z", completed_at: "2026-10-23T00:00:01Z", schema_version: "batch-integrity-v1",
    items: [{ filename: "<img src=x>.md", document_id: "d", inspection_status: "inspected" },
      { filename: "duplicate.md", document_id: "d", inspection_status: "inspected", reused: true },
      { filename: "empty.txt", inspection_status: "not_applicable", reason: "rejected_no_document" }],
    documents: { d: { filename: "source.md", findings: [{ severity: "unknown" }], scope: { complete: false, content_complete: true, sources_complete: true } } }, ...overrides,
  }, () => {});
  return host;
}

test("batch diagnosis preserves input rows, counts unique documents and safely marks partial/unknown", () => {
  const host = render();
  assert.equal(host.querySelectorAll("td").length, 9);
  assert.equal(host.querySelectorAll("details").length, 1);
  assert.equal(host.querySelector("img"), null);
  assert.match(host.textContent, /实际检查唯一文档 1/);
  assert.match(host.textContent, /复用本次同一来源检查/);
  assert.match(host.textContent, /拒收文件没有索引/);
  assert.match(host.textContent, /部分检查/);
});

test("changed snapshots and deferred inputs never appear as a pass", () => {
  const host = render({ changed_during_inspection: true, items: [{ filename: "late.pdf", inspection_status: "deferred", reason: "batch_budget_exhausted" }], documents: {} });
  assert.match(host.textContent, /发生变化/);
  assert.match(host.textContent, /达到本批检查预算/);
  assert.doesNotMatch(host.textContent, /质量通过率：|100%/);
});

test("unique document details render lazily and source actions retain their document", async () => {
  globalThis.document = new JSDOM("<main></main>").window.document;
  const host = document.querySelector("main"); let selected;
  const doc = { filename:"evidence.md", note:"机械检查", schema_version:"v1", snapshot_sha256:"abc",
    scope:{complete:true,content_complete:true,sources_complete:true,inspected:{blocks:1,chunks:1},total:{blocks:1,chunks:1}},
    observations:{token_budget:{observed:0,unknown:1},page_coverage:{status:"not_applicable"}},
    findings:[{code:"invalid_chunk_range",severity:"error",count:1,message:"区间无效",examples:[{chunk_id:"c",ordinal:0}]}] };
  renderBatchDiagnostics(host,{note:"只读",summary:{n_inputs:1,n_unique_inspected_documents:1,inspection_status_counts:{inspected:1}},
    budget:{read_bytes:100},items:[{document_id:"d",filename:"evidence.md",inspection_status:"inspected"}],documents:{d:doc},
    started_at:"2026-10-23T00:00:00Z",completed_at:"2026-10-23T00:00:01Z"},(id,example)=>{selected={id,example};});
  assert.equal(host.querySelectorAll("button").length,1);
  const details = host.querySelector("details");
  const opened = new Promise(resolve=>details.addEventListener("toggle",resolve,{once:true}));
  details.querySelector("summary").click();
  await opened;
  [...host.querySelectorAll("button")].find(button=>button.textContent==="查看切片列表").click();
  assert.equal(selected.id,"d"); assert.equal(selected.example.chunk_id,"c");
});
