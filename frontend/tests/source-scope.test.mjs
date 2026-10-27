import assert from "node:assert/strict";
import {test} from "node:test";
import {JSDOM} from "jsdom";
import {renderRetrievalEvidence} from "../retrieval-view.js";

test("source filtering is visible for keyword/blocked queries and untrusted names stay inert",()=>{
  const dom=new JSDOM("<div></div>");globalThis.document=dom.window.document;
  try{
    const host=document.createElement("div");
    for(const status of ["filtered","unavailable","ambiguous"]){
      renderRetrievalEvidence(host,{query_plan:{status:"not_requested",source_scope:{
        version:"literal-document-scope-v1",status,documents:[{filename:'<img src=x onerror="bad()">.pdf'}],
        unknown_filenames:["missing.pdf"],excluded_filenames:["other.md"]}}});
      assert.match(host.textContent,/检索来源范围/);
      assert.match(host.textContent,/数量限制之前/);
      assert.match(host.textContent,/不保证相关或回答正确/);
      assert.equal(host.querySelector("img,script,a"),null);
    }
  }finally{dom.window.close();delete globalThis.document;}
});
