import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { renderSegments } from "../segment-view.js";

test("inspector renders actual text, statistics and page provenance without executing markup", () => {
  globalThis.document = new JSDOM("<main></main>").window.document;
  const host = document.querySelector("main"); let selected;
  renderSegments(host, { view: "chunks", document_status: "processed", statistics: { count: 3, min_characters: 4, max_characters: 800, average_characters: 200 },
    items: [{ ordinal: 1, kind: "chunk", characters: 24, content: "<script>attack()</script>\n第二行", section: "<img src=x>", source_spans: [{ page_number: 2, block_key: "#/texts/4" }] }] }, (page) => { selected = page; });
  assert.equal(host.querySelectorAll("script,img").length, 0);
  assert.match(host.textContent, /#2 · 索引切片/);
  assert.equal(host.querySelector("pre").textContent, "<script>attack()</script>\n第二行");
  host.querySelector("button").click(); assert.equal(selected, 2);
});

test("old block indexes and failed rebuilds are labeled without pretending active completeness", () => {
  globalThis.document = new JSDOM("<main></main>").window.document;
  const host = document.querySelector("main");
  renderSegments(host, { view: "blocks", document_status: "failed", statistics: { count: 0 }, items: [] }, () => {});
  assert.match(host.textContent, /不参与当前检索/);
  assert.match(host.textContent, /旧索引需要重新解析/);
});

test("cross-page fragments expose both source pages and the complete input estimate", () => {
  globalThis.document = new JSDOM("<main></main>").window.document;
  const host = document.querySelector("main"); const selected = [];
  renderSegments(host, { view: "chunks", document_status: "processed", statistics: { count: 1 }, items: [
    { ordinal: 0, kind: "chunk", content: "跨页内容", characters: 4,
      processing_metadata: { kind: "prose", input_tokens: 496, fragmented: true },
      source_spans: [{ page_number: 2 }, { page_number: 3 }] }
  ] }, (page) => selected.push(page));
  assert.match(host.textContent, /完整输入 496 估算 token · 超长元素分片/);
  for (const button of host.querySelectorAll("button")) button.click();
  assert.deepEqual(selected, [2, 3]);
});

test("PDF logical cell mappings show saved offsets and merged identities as safe text", () => {
  globalThis.document = new JSDOM("<main></main>").window.document;
  const host = document.querySelector("main");
  renderSegments(host, {view:"chunks",document_status:"processed",statistics:{count:1},items:[{
    ordinal:0,kind:"chunk",characters:16,content:"A: <img src=x>",source_spans:[{
      source_mapping:"docling-rendered-row-v1",row_number:2,page_number:4,
      table_cells:[{role:"value",column_number:1,chunk_range:[3,14],text_range:[20,31],
        source_cells:[{index:3,rows:[2,2],columns:[1,2]}]}]
    }]}]}, () => {});
  assert.match(host.textContent, /PDF 表格来源映射/);
  assert.match(host.textContent, /原页仍高亮整表/);
  assert.match(host.textContent, /R2–2 C1–2/);
  assert.match(host.textContent, /<img src=x>/);
  assert.equal(host.querySelectorAll("img").length,0);
});
