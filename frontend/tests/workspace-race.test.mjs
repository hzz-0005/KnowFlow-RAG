import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {test} from "node:test";
import {JSDOM} from "jsdom";

test("switching knowledge bases clears old actions and discards late responses", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const source = await readFile(new URL("../app.js", import.meta.url), "utf8");
  const dom = new JSDOM(html, {url:"http://localhost:8080/", runScripts:"outside-only"});
  const {window} = dom;
  window.Headers = Headers;
  const kbs = [{id:"a",name:"Alpha"},{id:"b",name:"Beta"}];
  const pending = new Map();
  const payload = (value) => ({ok:true,status:200,json:async()=>value});
  window.fetch = async (url) => {
    if (url === "/api/v1/knowledge-bases") return payload(kbs);
    return new Promise(resolve=>pending.set(url, value=>resolve(payload(value))));
  };
  const stripped = source.replace(/^import[^\n]*\n/, "");
  window.eval(`${stripped}\nglobalThis.harness = {state,refreshWorkspace};`);
  const {state,refreshWorkspace} = window.harness;
  state.token = "dummy-local-test-token";
  state.knowledgeBases = kbs; state.currentKnowledgeBase = kbs[0];
  const alpha = refreshWorkspace();
  const tick = () => new Promise(resolve=>setImmediate(resolve));
  await tick();
  // Simulate previously loaded actions while an old refresh remains in flight.
  window.document.querySelector("#document-list").innerHTML = '<button>old Alpha delete</button>';
  window.document.querySelector("#answer-panel").textContent = 'old Alpha answer';
  state.lastAnswer = {citations:[{document_id:'old-alpha'}]};
  window.document.querySelector('[data-nav-action="select"][data-kb-id="b"]').click();
  assert.equal(window.document.querySelector("#document-list").textContent, "");
  assert.equal(window.document.querySelector("#answer-panel").textContent, "");
  assert.equal(state.lastAnswer, null);
  assert.equal(window.document.querySelector("#open-quality").disabled, true);
  await tick();
  const doc = (id) => [{id,filename:`${id}.pdf`,file_size:10,status:"processed",
    graph_status:"not_requested",created_at:"2026-10-23T00:00:00Z"}];
  pending.get("/api/v1/knowledge-bases/b/documents")(doc("beta"));
  pending.get("/api/v1/knowledge-bases/b/evaluations")([]);
  pending.get("/api/v1/knowledge-bases/b/ingestion-batches")([{id:"beta-batch",active:0}]);
  await tick();
  assert.match(window.document.querySelector("#document-list").textContent, /beta.pdf/);
  pending.get("/api/v1/knowledge-bases/a/documents")(doc("alpha"));
  pending.get("/api/v1/knowledge-bases/a/evaluations")([]);
  pending.get("/api/v1/knowledge-bases/a/ingestion-batches")([{id:"alpha-batch",active:1}]);
  await alpha; await tick();
  assert.match(window.document.querySelector("#document-list").textContent, /beta.pdf/);
  assert.doesNotMatch(window.document.querySelector("#document-list").textContent, /alpha.pdf/);
  assert.equal(state.contentKnowledgeBaseId, "b");
  assert.equal(state.batches[0].id, "beta-batch");
  dom.window.close();
});
