import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { renderDocumentReferences, renderReferenceExpansion } from "../reference-view.js";
function host() { globalThis.document = new JSDOM("<main></main>").window.document; return document.querySelector("main"); }
test("ambiguous references retain candidate source pages without silently selecting a target", () => {
  const root = host(); let page;
  renderDocumentReferences(root, {document_status:"processed", report:{total:1,statuses:{ambiguous:1}}, items:[{source_quote:"第2节",status:"ambiguous",source:{block_key:"src",kind:"text",text:"<script>x</script>",source_spans:[{page_number:8}]},targets:[{block_key:"a",kind:"section_header",text:"2 A",source_spans:[{page_number:2}]},{block_key:"b",kind:"section_header",text:"2 B",source_spans:[{page_number:5}]}]}]}, (n)=>{page=n;});
  assert.equal(root.querySelector("script"),null);
  assert.match(root.textContent,/同号歧义/); assert.match(root.textContent,/不会自动展开/);
  root.querySelector("button").click(); assert.equal(page,8);
  assert.equal(root.querySelectorAll("button").length,3);
});
test("expansion diagnostics expose additions, budgets, unknown index and unresolved labels", () => {
  const root=host(); renderReferenceExpansion(root,{enabled:true,base_count:1,added_count:2,max_hops:2,truncated:true,note:"仅结构引用",inferred_section_scopes:["section:2"],unindexed_documents:["<img>"],unresolved:[{filename:"demo.pdf",label:"section:2",status:"ambiguous"}]});
  assert.match(root.textContent,/补充 2 个片段/); assert.match(root.textContent,/同号歧义/);
  assert.match(root.textContent,/超过展开预算/); assert.equal(root.querySelector("img"),null);
  assert.match(root.textContent,/按明确编号区间补充候选/);
});

test("recovered table captions expose exact source text and a combined page check", () => {
  const root = host(); let observed;
  const span = { page_number: 8, block_key: "caption", bbox: {left:0.1,top:0.1,right:0.4,bottom:0.2} };
  renderDocumentReferences(root, { document_status:"processed", report:{total:1,statuses:{resolved:1},recovered_table_targets:["table:5"]}, items:[{source_quote:"Table 5",status:"resolved",kind:"table",targets:[{block_key:"table",kind:"table",text:"Orion: 83",source_spans:[{page_number:8}],caption_recovery:{text:"Table 5: <img> comparison",source_span:span}}]}]}, (page, spans) => {observed={page,spans};});
  assert.match(root.textContent, /唯一相邻表格恢复/);
  assert.match(root.textContent, /Table 5: <img> comparison/);
  assert.equal(root.querySelector("img"), null);
  [...root.querySelectorAll("button")].find(button => button.textContent.includes("核对恢复表注")).click();
  assert.equal(observed.page, 8); assert.equal(observed.spans.length, 2);
});
