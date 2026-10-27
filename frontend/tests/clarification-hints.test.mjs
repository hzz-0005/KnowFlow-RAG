import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {test} from "node:test";
import {JSDOM} from "jsdom";

test("clarification availability is escaped and survives chat reload without selection or auto-send", async()=>{
  const html=await readFile(new URL("../index.html",import.meta.url),"utf8");
  const source=await readFile(new URL("../app.js",import.meta.url),"utf8");
  const dom=new JSDOM(html,{url:"http://localhost:8080/",runScripts:"outside-only"});
  const {window}=dom;
  const calls=[];
  window.Headers=Headers;
  window.HTMLElement.prototype.scrollIntoView=()=>{};
  window.renderRichAnswer=async(node,text)=>{node.textContent=text;};
  window.renderRetrievalEvidence=()=>{};
  const hints={version:"clarification-hints-v1",kind:"available_documents_not_relevance_ranked",
    has_more:true,documents:[{document_id:"doc",filename:'<img src=x onerror="alert(1)">.pdf'}]};
  const saved={id:"turn",sequence:1,question:"它呢？",status:"needs_clarification",
    response:{answer:"此前引用的记忆已停用。请明确对象。",refused:true,strategy:"keyword",citations:[],clarification:hints}};
  window.fetch=async(url,options)=>{
    calls.push([url,options?.method||"GET"]);
    return {ok:true,status:200,json:async()=>({id:"chat",revision:1,turns:[saved],has_more:false})};
  };
  window.eval(source.replace(/^import[^\n]*\n/,"")+"\nglobalThis.hintsTest={state,loadConversation,renderAnswer};");
  const h=window.hintsTest;
  h.state.token="offline";h.state.currentKnowledgeBase={id:"kb",name:"KB"};
  try{
    await h.loadConversation("chat",h.state.conversationEpoch);
    const box=window.document.querySelector("#conversation-history .clarification-hints");
    assert.match(box.textContent,/未按相关性筛选/);
    assert.match(box.textContent,/前 5 份/);
    assert.match(box.textContent,/<img/);
    assert.equal(box.querySelector("img"),null);
    assert.equal(box.querySelector("button, a, input"),null);
    assert.match(window.document.querySelector("#conversation-history").textContent,/记忆已停用/);
    assert.ok(calls.every(([,method])=>method==="GET"));
    await h.renderAnswer(saved.response);
    assert.match(window.document.querySelector("#answer-panel .clarification-hints").textContent,/<img/);
    assert.equal(window.document.querySelector("#answer-panel .clarification-hints img"),null);
  }finally{dom.window.close();}
});
