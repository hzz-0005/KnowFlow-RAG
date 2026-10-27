import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {test} from "node:test";
import {JSDOM} from "jsdom";

test("source context is separate, closed by default and escaped, with original witness intact", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const source = await readFile(new URL("../app.js", import.meta.url), "utf8");
  const dom = new JSDOM(html, {url:"http://localhost:8080/", runScripts:"outside-only"});
  const {window} = dom;
  window.fetch = async () => ({ok:true,status:200,json:async()=>[]});
  window.Headers = Headers;
  window.tabularLocations = () => [];
  window.renderRetrievalEvidence = () => {};
  window.renderRichAnswer = async (node, text) => { node.textContent=text; };
  window.HTMLElement.prototype.scrollIntoView = () => {};
  window.eval(source.replace(/^import[^\n]*\n/, "") + "\nglobalThis.renderAnswerForTest=renderAnswer;");
  await window.renderAnswerForTest({strategy:"keyword", answer:"A statement.[1]", citations:[{
    chunk_id:"test", document_id:"doc", filename:"manual.txt", quote:"VERIFIED WITNESS AT END",
    context_excerpt:'Other source text <img src=x onerror="alert(1)">', source_spans:[],
  }]});
  const panel = window.document.querySelector("#answer-panel");
  assert.equal(panel.querySelector(".citation > .citation-quote").textContent, "VERIFIED WITNESS AT END");
  const context = panel.querySelector(".citation-context");
  assert.equal(context.open, false);
  assert.match(context.textContent, /不表示.*逐句审核/);
  assert.match(context.querySelector("blockquote").textContent, /<img/);
  assert.equal(context.querySelector("img"), null);
  dom.window.close();
});
