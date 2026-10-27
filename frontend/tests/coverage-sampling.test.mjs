import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {test} from "node:test";
import {JSDOM} from "jsdom";

test("file sampling and citation counts cannot appear as full-content answer coverage", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const source = await readFile(new URL("../app.js", import.meta.url), "utf8");
  const dom = new JSDOM(html, {url:"http://localhost:8080/",runScripts:"outside-only"});
  dom.window.Headers=Headers; dom.window.fetch=async()=>({ok:true,json:async()=>[]});
  try {
    dom.window.eval(source.replace(/^import[^\n]*\n/, "")+"\nglobalThis.coverageView=coverageHtml;");
    const host=dom.window.document.createElement("div");
    host.innerHTML=dom.window.coverageView({mode:"sweep",checked:20,eligible_documents:21,
      omitted_documents:1,cited_documents:2,documents:[
        {filename:"<script>bad</script>.md",evidence_chunks:2,cited_in_answer:false},
      ]});
    assert.match(host.textContent,/片段抽样 20\/21/);
    assert.match(host.textContent,/答案引用 2 份/);
    assert.match(host.textContent,/1 份未检查/);
    assert.match(host.textContent,/非全文检查/);
    assert.doesNotMatch(host.textContent,/答案覆盖|未提及/);
    assert.equal(host.querySelectorAll("script").length,0);
    assert.match(host.querySelector("[title]").title,/不能判断内容是否缺失/);
    host.innerHTML=dom.window.coverageView({mode:"sweep",checked:5,answered:1,documents:[]});
    assert.match(host.textContent,/5\/5/);
    assert.match(host.textContent,/非全文检查/);
  } finally { dom.window.close(); }
});

test("a URL fragment cannot replace the signed-in account token", async () => {
  const html=await readFile(new URL("../index.html",import.meta.url),"utf8");
  const source=await readFile(new URL("../app.js",import.meta.url),"utf8");
  const dom=new JSDOM(html,{url:"http://localhost:8080/#auth=untrusted.token.value",runScripts:"outside-only"});
  dom.window.Headers=Headers; dom.window.fetch=async()=>({ok:true,json:async()=>[]});
  try {
    dom.window.eval(source.replace(/^import[^\n]*\n/, ""));
    assert.equal(dom.window.sessionStorage.getItem("knowflow_access_token"),null);
  } finally { dom.window.close(); }
});
