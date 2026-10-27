import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { renderParsingReport, renderSourcePage } from "../source-view.js";

function host() {
  const dom = new JSDOM("<main></main>");
  globalThis.document = dom.window.document;
  return document.querySelector("main");
}

test("old indexes have an explicit unknown report, not a fabricated success", () => {
  const root = host(); renderParsingReport(root, {}, () => {});
  assert.match(root.textContent, /尚无解析报告/);
  assert.doesNotMatch(root.textContent, /结构解析完成/);
});

test("report preserves warnings, unknown OCR certainty, and page actions without HTML injection", () => {
  const root = host(); let selected;
  renderParsingReport(root, { parsing_report: { status: "needs_review", parser: "docling", ocr_requested: true,
    warnings: ["<img src=x onerror=alert(1)>"], pages: [{ page_number: 2, native_characters: 0, indexed_characters: 20 }] } }, (n) => { selected = n; });
  assert.equal(root.querySelector("img"), null);
  assert.match(root.textContent, /仍需与原页核对/);
  root.querySelector("button").click(); assert.equal(selected, 2);
});

test("source overlays only validated normalized boxes on the selected page", () => {
  const root = host();
  renderSourcePage(root, "blob:test", 2, [
    { page_number: 2, bbox: { left: .1, top: .2, right: .5, bottom: .6 } },
    { page_number: 1, bbox: { left: .1, top: .2, right: .5, bottom: .6 } },
    { page_number: 2, bbox: { left: -2, top: 0, right: 1, bottom: 1 } },
  ]);
  assert.equal(root.querySelectorAll(".source-highlight").length, 1);
  assert.equal(root.querySelector(".source-highlight").style.left, "10%");
  assert.equal(root.querySelector("img").alt, "文档原页，第 2 页");
});

test("report distinguishes estimated budgets and reuse from OCR accuracy", () => {
  const root = host();
  renderParsingReport(root, { parsing_report: { parser: "docling", status: "completed", pages: [],
    chunking: { strategy: "structure_tokens", chunk_count: 12, cross_page_chunks: 3,
      max_input_tokens: 496, max_tokens: 512, token_reserve: 16, note: "来源覆盖检查不代表语义正确。",
      source_mapping_revision: "piece-relative-ranges-v1" },
    parse_artifact: { available: true, reused: true } } }, () => {});
  assert.match(root.textContent, /12 个 · 3 个跨页块/);
  assert.match(root.textContent, /496 \/ 512 估算 token/);
  assert.match(root.textContent, /没有重新 OCR/);
  assert.match(root.textContent, /不代表语义正确/);
  assert.match(root.textContent, /片段内相对区间/);
  assert.match(root.textContent, /无字符映射的表格仍保留未知/);
  renderParsingReport(root, { parsing_report: { parser: "docling", ocr_requested: true,
    comparison_copy: { reused_parser_and_vectors: true }, parse_artifact: { available: true } } }, () => {});
  assert.match(root.textContent, /历史解析记录曾请求 OCR/);
  assert.match(root.textContent, /本次对照没有重新 OCR/);
  assert.match(root.textContent, /仅重算来源区间/);
});
