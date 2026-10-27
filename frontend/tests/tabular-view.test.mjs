import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { renderSegments } from "../segment-view.js";
import { renderParsingReport, tabularLocations } from "../source-view.js";

function root() {
  globalThis.document = new JSDOM("<main></main>").window.document;
  return document.querySelector("main");
}

test("tabular report discloses hidden sheets, unverified caches and no invented PDF pages", () => {
  const host = root();
  renderParsingReport(host, { parsing_report: { parser: "tabular", source_format: "xlsx",
    status: "needs_review", sheet_count: 1, table_count: 1, pages: [],
    sheets: [{ sheet_name: "<img src=x>", nonempty_rows: 3, column_count: 4, sheet_state: "hidden", merged_ranges: ["A1:B1"] }],
    tabular: { formula_cells_missing_cache: 1, formula_cells_cached: 2, formula_cells_cached_error: 1 } } }, () => assert.fail("no PDF"));
  assert.match(host.textContent, /隐藏工作表（已纳入索引）/);
  assert.match(host.textContent, /缺失 1 个.*未验证 2 个.*错误 1 个/);
  assert.match(host.textContent, /无固定页码/);
  assert.doesNotMatch(host.textContent, /0 页/);
  assert.equal(host.querySelector("img"), null);
  assert.equal(host.querySelector("button"), null);
});

test("cell coordinates are deduplicated independently of page spans", () => {
  const span = { source_type: "spreadsheet", sheet_name: "库存", cell_range: "A2:C2" };
  assert.deepEqual(tabularLocations([span, span, { page_number: 1 }]), ["库存 · A2:C2"]);
  assert.deepEqual(tabularLocations(), []);
});

test("cell inspector lazily paginates literal values and distinguishes missing/error/zero without markup", async () => {
  const host = root(); const calls = [];
  const cellRows = [
    { coordinate: "A2", raw_value: "<img src=x>", formula_cache_status: "not_formula" },
    { coordinate: "B2", raw_value: "=0", formula: "=0", cached_value: 0, formula_cache_status: "available_unverified" },
    { coordinate: "C2", raw_value: null, formula: "=B2", formula_cache_status: "missing", merged_from: "A2", row_hidden: true },
    { coordinate: "D2", raw_value: "=1/0", formula: "=1/0", cached_value: "#DIV/0!", formula_cache_status: "cached_error" },
  ];
  renderSegments(host, { view: "blocks", document_status: "processed", statistics: { count: 1 }, items: [
    { ordinal: 0, block_key: "sheet-0-table-0", kind: "table", characters: 4, content: "test",
      tabular_source: { sheet_name: "库存", cell_count: 4 }, source_spans: [
        { source_type: "spreadsheet", sheet_name: "库存", cell_range: "A2:D2" }] }
  ] }, () => assert.fail("no pages"), async (key, offset) => {
    calls.push([key, offset]); return { total: 4, offset, limit: 2, cells: cellRows.slice(offset, offset + 2) };
  });
  assert.equal(calls.length, 0);
  const details = host.querySelector("details"); details.open = true;
  details.dispatchEvent(new document.defaultView.Event("toggle"));
  await new Promise(resolve => setTimeout(resolve, 5));
  assert.equal(host.querySelector("img"), null);
  assert.match(host.textContent, /0（未验证）/);
  assert.equal(host.querySelectorAll("button")[0].disabled, true);
  [...host.querySelectorAll("button")].find(n => n.textContent === "下一批单元格").click();
  await new Promise(resolve => setTimeout(resolve, 5));
  assert.match(host.textContent, /未知，未计算/);
  assert.match(host.textContent, /公式缓存错误：#DIV\/0!/);
  assert.match(host.textContent, /合并自 A2/);
  assert.match(host.textContent, /隐藏行（已纳入）/);
  assert.deepEqual(calls, [["sheet-0-table-0", 0], ["sheet-0-table-0", 2]]);
});
