import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {test} from "node:test";
import {JSDOM} from "jsdom";

test("prepared or failed summary input never appears as a completed full read",async()=>{
  const html=await readFile(new URL("../index.html",import.meta.url),"utf8");
  const source=await readFile(new URL("../app.js",import.meta.url),"utf8");
  const dom=new JSDOM(html,{url:"http://localhost:8080/",runScripts:"outside-only"});
  dom.window.Headers=Headers;dom.window.fetch=async()=>({ok:true,json:async()=>[]});
  try{
    dom.window.eval(source.replace(/^import[^\n]*\n/,"")+"\nglobalThis.summaryView=summaryScopeHtml;");
    const host=dom.window.document.createElement("div");
    host.innerHTML=dom.window.summaryView({document:{filename:"long.pdf"},total_chunks:138,read_chunks:0,coverage:"parsed_text_prepared_not_generated",mode:"whole_document"});
    assert.match(host.textContent,/未确认模型读取/);
    assert.match(host.textContent,/已读 0\/138/);
    assert.doesNotMatch(host.textContent,/全文直读/);
    host.innerHTML=dom.window.summaryView({document:{filename:"long.pdf"},total_chunks:138,read_chunks:99,coverage:"partial_parsed_text_segmented",mode:"segmented_map_reduce",segments:4,summarized_segments:3});
    assert.match(host.textContent,/39 个切片未读或未形成可用笔记/);
  }finally{dom.window.close();}
});
