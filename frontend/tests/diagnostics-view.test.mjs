import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { renderDiagnostics } from "../diagnostics-view.js";

function render(overrides = {}, onInspect = () => {}) {
  globalThis.document = new JSDOM("<main></main>").window.document;
  const host = document.querySelector("main");
  renderDiagnostics(host, {
    note: "机械检查，不代表语义正确", schema_version: "v1", snapshot_sha256: "abc",
    scope: { complete: false, content_complete: true, inspected: { blocks: 1, chunks: 1 }, total: { blocks: 5, chunks: 8 } },
    observations: { token_budget: { observed: 0, unknown: 1, max_input_tokens: null, saved_budget: null }, page_coverage: { status: "unknown" } },
    findings: [], ...overrides,
  }, onInspect);
  return host;
}

test("partial scope and unavailable budgets remain explicit instead of displaying zero or passed", () => {
  const host = render();
  assert.match(host.textContent, /部分检查/);
  assert.match(host.textContent, /最高 未知 \/ 未知/);
  assert.doesNotMatch(host.textContent, /100%|全量通过/);
});

test("locations are plain text, examples have counts, and links target the selected evidence", () => {
  let chosen;
  const example = { chunk_id: "c2", ordinal: 7, block_key: "<img src=x>", sheet_name: "sheet", cell_range: "A2:B2" };
  const host = render({ findings: [{ severity: "error", count: 19, message: "<script>oops</script>", examples: [example] }] }, (value) => { chosen = value; });
  assert.equal(host.querySelector("img,script"), null);
  assert.match(host.textContent, /展示 1 \/ 19/);
  assert.match(host.textContent, /切片 #8/);
  host.querySelector("button").click();
  assert.deepEqual(chosen, example);
});

test("spreadsheet diagnosis uses cell locations and does not invent PDF coverage", () => {
  const host = render({ observations: { token_budget: { observed: 1, unknown: 0, max_input_tokens: 40, saved_budget: 100 }, page_coverage: { status: "not_applicable" } } });
  assert.match(host.textContent, /工作表与单元格/);
  assert.doesNotMatch(host.textContent, /页：0/);
});

test("withheld sources keep a row-complete inspection labeled partial", () => {
  const host = render({ scope: { complete: true, content_complete: true, sources_complete: false,
    inspected: { blocks: 1, chunks: 1 }, total: { blocks: 1, chunks: 1 } } });
  assert.match(host.textContent, /部分检查/);
  assert.doesNotMatch(host.textContent, /已读全量记录/);
});
